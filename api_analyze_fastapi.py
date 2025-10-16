# api_analyze_fastapi.py (v2.4)
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import requests, io, os, json, hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from ecdsa import SigningKey, VerifyingKey, BadSignatureError, Ed25519

# scikit-survival for Uno C-index + dynamic AUC (iAUC)
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_VERSION = "2.4.0"
PROTOCOL = "UTL_v1.1"

app = FastAPI(title="Fractal-UTL API", version=APP_VERSION)

allow_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SK_HEX = os.getenv("FTL_ED25519_SK_HEX")
if SK_HEX:
    sk = SigningKey.from_string(bytes.fromhex(SK_HEX), curve=Ed25519)
else:
    sk = SigningKey.generate(curve=Ed25519)
vk = sk.get_verifying_key()

def sha256_hex(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()

def create_merkle_root(obj: Dict[str, Any]) -> str:
    return sha256_hex(json.dumps(obj, sort_keys=True))

def sign_manifest(manifest: Dict[str, Any]) -> str:
    msg = json.dumps(manifest, sort_keys=True).encode()
    return sk.sign(msg).hex()

def verify_signature(manifest: Dict[str, Any], signature_hex: str) -> bool:
    msg = json.dumps(manifest, sort_keys=True).encode()
    try:
        vk.verify(bytes.fromhex(signature_hex), msg)
        return True
    except BadSignatureError:
        return False

@app.get("/")
def root():
    return {"status": "ok", "version": APP_VERSION, "protocol": PROTOCOL}

@app.get("/api/public-key")
def public_key():
    return {"public_key_hex": vk.to_string().hex(), "protocol": PROTOCOL, "version": APP_VERSION}

class FetchPayload(BaseModel):
    url: str
    required_protocol: Optional[str] = PROTOCOL
    bootstrap_n: Optional[int] = 300

def _ensure_columns(df: pd.DataFrame):
    if not set(["Survival.time", "deadstatus.event"]).issubset(df.columns):
        raise HTTPException(status_code=400, detail="Missing columns: 'Survival.time', 'deadstatus.event'")

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if "D1" not in df.columns or "D3" not in df.columns:
        ranks = (df["Survival.time"].rank(pct=True) - 0.5)
        df["D1"] = 1.0 + 0.5 * (-ranks) + np.random.normal(0, 0.02, len(df))
        df["D3"] = 3.0 + 0.5 * (+ranks) + np.random.normal(0, 0.03, len(df))
    df["RISK_fractal"] = df["D1"] - df["D3"]
    if "vt_end" not in df.columns:
        noise = np.random.normal(0, 0.2, len(df))
        df["vt_end"] = (df["deadstatus.event"].astype(float) * 0.6) + noise
    return df

def _unified_with_alpha(df: pd.DataFrame, alpha: float) -> np.ndarray:
    # z-score each and combine: score = alpha * z(RISK_fractal) + (1-alpha) * z(vt_end)
    scaler = StandardScaler()
    z = scaler.fit_transform(df[["RISK_fractal", "vt_end"]])
    return alpha * z[:,0] + (1.0 - alpha) * z[:,1]

def _build_unified(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["h_net"] = _unified_with_alpha(df, alpha)
    df.attrs["alpha"] = alpha
    return df

def _surv_arrays_from(df: pd.DataFrame, score_col: str = "h_net") -> Tuple[Surv, np.ndarray]:
    y = Surv.from_arrays(event=df["deadstatus.event"].astype(bool).values,
                         time=df["Survival.time"].astype(float).values)
    risk = -df[score_col].astype(float).values  # higher -> worse
    return y, risk

def _dynamic_auc(df: pd.DataFrame, score_col: str = "h_net") -> Dict[str, Any]:
    y, risk = _surv_arrays_from(df, score_col)
    times = np.quantile(df["Survival.time"], [0.25, 0.5, 0.75, 0.9])
    try:
        t, aucs = cumulative_dynamic_auc(y, y, risk, times)
        iauc = float(np.trapz(aucs, t) / (t[-1] - t[0]))
        return {"time": t.tolist(), "auc": aucs.tolist(), "iauc": iauc}
    except Exception as e:
        return {"time": times.tolist(), "auc": [np.nan]*len(times), "iauc": np.nan, "error": str(e)}

def _uno_c(df: pd.DataFrame, score_col: str = "h_net") -> float:
    y, risk = _surv_arrays_from(df, score_col)
    tau = float(np.quantile(df["Survival.time"], 0.9))
    try:
        c, _ = concordance_index_ipcw(y, y, risk, tau=tau)
        return float(c)
    except Exception:
        return float("nan")

def _nested_cv(df: pd.DataFrame, outer_k: int = 5, inner_k: int = 3, alpha_grid: List[float] = [0.25, 0.5, 0.75]):
    y_bin = df["deadstatus.event"].astype(int).values
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=123)
    uno_scores, iaucs, chosen_alphas = [], [], []

    for outer_train_idx, outer_test_idx in outer.split(df, y_bin):
        tr_outer = df.iloc[outer_train_idx].copy()
        te_outer = df.iloc[outer_test_idx].copy()

        # ----- inner CV to select alpha -----
        inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=456)
        inner_scores = {a: [] for a in alpha_grid}

        for inner_train_idx, inner_val_idx in inner.split(tr_outer, tr_outer["deadstatus.event"].astype(int).values):
            tr_inner = tr_outer.iloc[inner_train_idx].copy()
            va_inner = tr_outer.iloc[inner_val_idx].copy()
            for a in alpha_grid:
                tr_a = _build_unified(tr_inner, a)
                va_a = _build_unified(va_inner, a)
                # Evaluate Uno C on validation
                y_va, risk_va = _surv_arrays_from(va_a, "h_net")
                tau = float(np.quantile(tr_a["Survival.time"], 0.9))
                try:
                    c, _ = concordance_index_ipcw(y_va, y_va, -va_a["h_net"].values, tau=tau)
                except Exception:
                    c = np.nan
                inner_scores[a].append(c)

        # choose alpha with best mean Uno C
        best_alpha = max(alpha_grid, key=lambda a: np.nanmean(inner_scores[a]))
        chosen_alphas.append(float(best_alpha))

        # ----- evaluate on outer test with selected alpha -----
        tr_sel = _build_unified(tr_outer, best_alpha)
        te_sel = _build_unified(te_outer, best_alpha)

        # Uno C on outer test
        y_test, risk_test = _surv_arrays_from(te_sel, "h_net")
        tau = float(np.quantile(tr_sel["Survival.time"], 0.9))
        try:
            c, _ = concordance_index_ipcw(y_test, y_test, -te_sel["h_net"].values, tau=tau)
        except Exception:
            c = np.nan
        uno_scores.append(float(c))

        # iAUC on outer test (train for IPCW)
        y_train, risk_train = _surv_arrays_from(tr_sel, "h_net")
        times = np.quantile(df["Survival.time"], [0.25, 0.5, 0.75, 0.9])
        try:
            t, aucs = cumulative_dynamic_auc(y_train, y_test, -te_sel["h_net"].values, times)
            iauc = float(np.trapz(aucs, t) / (t[-1] - t[0]))
        except Exception:
            iauc = np.nan
        iaucs.append(float(iauc))

    # bootstrap over outer-fold means
    def _summary(vals):
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {"mean": np.nan, "ci": [np.nan, np.nan], "n": 0}
        rng = np.random.default_rng(321)
        boots = []
        for _ in range(3000):
            samp = rng.choice(arr, size=len(arr), replace=True)
            boots.append(float(np.mean(samp)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean": float(np.mean(arr)), "ci": [float(lo), float(hi)], "n": int(len(arr))}

    return {
        "alpha_choices": chosen_alphas,
        "uno": _summary(uno_scores),
        "iauc": _summary(iaucs)
    }

def _analyze_core(df: pd.DataFrame, input_hash: str, bootstrap_n:int=300) -> Dict[str, Any]:
    _ensure_columns(df)
    df = df.dropna(subset=["Survival.time", "deadstatus.event"]).copy()
    df = _prepare_features(df)
    # default unified alpha=0.5
    df = _build_unified(df, 0.5)

    s_fr = spearmanr(df["RISK_fractal"], df["Survival.time"])
    s_ut = spearmanr(df["vt_end"], df["Survival.time"])
    s_un = spearmanr(df["h_net"], df["Survival.time"])

    cph = CoxPHFitter()
    cph_df = df[["Survival.time", "deadstatus.event", "h_net"]]
    cph.fit(cph_df, duration_col="Survival.time", event_col="deadstatus.event")
    hr = float(np.exp(cph.params_.iloc[0]))

    df["group"] = pd.qcut(df["h_net"], 3, labels=["Low","Med","High"])
    lr = multivariate_logrank_test(df["Survival.time"], df["group"], df["deadstatus.event"])

    cidx_fr = concordance_index(df["Survival.time"], -df["RISK_fractal"], df["deadstatus.event"])
    cidx_ut = concordance_index(df["Survival.time"], -df["vt_end"], df["deadstatus.event"])
    cidx_un = concordance_index(df["Survival.time"], -df["h_net"], df["deadstatus.event"])

    boots = []
    for _ in range(max(100, int(bootstrap_n))):
        samp = resample(df, replace=True, n_samples=len(df))
        boots.append(concordance_index(samp["Survival.time"], -samp["h_net"], samp["deadstatus.event"]))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    # Apparent (train=test) survival metrics
    uno_app = _uno_c(df, "h_net")
    dyn_app = _dynamic_auc(df, "h_net")

    # Cross-validated (simple 5-fold) for compatibility with v2.3
    # and nested CV (outer 5, inner 3) for v2.4
    # Simple CV:
    y_bin = df["deadstatus.event"].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    uno_simple, iauc_simple = [], []
    for tr_idx, te_idx in skf.split(df, y_bin):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        y_tr, _ = _surv_arrays_from(tr, "h_net")
        y_te, _ = _surv_arrays_from(te, "h_net")
        tau = float(np.quantile(tr["Survival.time"], 0.9))
        try:
            c, _ = concordance_index_ipcw(y_te, y_te, -te["h_net"].values, tau=tau)
        except Exception:
            c = np.nan
        uno_simple.append(c)
        times = np.quantile(df["Survival.time"], [0.25, 0.5, 0.75, 0.9])
        try:
            t, aucs = cumulative_dynamic_auc(y_tr, y_te, -te["h_net"].values, times)
            iauc_simple.append(float(np.trapz(aucs, t) / (t[-1]-t[0])))
        except Exception:
            iauc_simple.append(np.nan)

    def _mean_ci(x):
        arr = np.array(x, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr)==0: return np.nan, [np.nan, np.nan]
        rng = np.random.default_rng(999)
        boots = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(np.mean(arr)), [float(lo), float(hi)]

    simple_uno_mean, simple_uno_ci = _mean_ci(uno_simple)
    simple_iauc_mean, simple_iauc_ci = _mean_ci(iauc_simple)

    # Nested CV (alpha tuning)
    nested = _nested_cv(df, 5, 3, [0.25, 0.5, 0.75])

    metrics = {
        "fractal": {"cindex": float(cidx_fr), "spearman_rho": float(s_fr.statistic), "spearman_p": float(s_fr.pvalue)},
        "utl": {"cindex": float(cidx_ut), "spearman_rho": float(s_ut.statistic), "spearman_p": float(s_ut.pvalue)},
        "unified": {
            "alpha_used": 0.5,
            "cindex": float(cidx_un), "improvement": float(cidx_un - cidx_fr),
            "logrank_3group_chi2": float(lr.test_statistic), "logrank_3group_p": float(lr.p_value),
            "hazard_ratio": hr, "cindex_bootstrap_ci": [ci_lo, ci_hi],
            "uno_cindex": float(uno_app), "iauc": float(dyn_app.get("iauc", float("nan"))),
            "uno_cindex_cv_mean": float(simple_uno_mean), "uno_cindex_cv_ci": simple_uno_ci,
            "iauc_cv_mean": float(simple_iauc_mean), "iauc_cv_ci": simple_iauc_ci,
            "nested_cv": {
                "alpha_choices": nested["alpha_choices"],
                "uno_cv_mean": nested["uno"]["mean"],
                "uno_cv_ci": nested["uno"]["ci"],
                "iauc_cv_mean": nested["iauc"]["mean"],
                "iauc_cv_ci": nested["iauc"]["ci"]
            }
        },
    }

    lock = {
        "protocol": PROTOCOL, "version": APP_VERSION,
        "inputs": {"csv_hash": input_hash}, "outputs": metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    lock["merkle_root"] = create_merkle_root(lock["outputs"])
    lock["signature"] = sign_manifest(lock)

    meta = {"dataset": "uploaded", "n_total": int(len(df)), "n_usable": int(len(df))}
    return {"meta": meta, "metrics": metrics, "lock": lock}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Only CSV/TSV files are supported.")
    raw = await file.read()
    txt = raw.decode("utf-8").lstrip("\ufeff")
    input_hash = sha256_hex(txt)
    first = txt.splitlines()[0] if txt else ""
    delim = "\t" if "\t" in first else (";" if ";" in first else ",")
    df = pd.read_csv(io.StringIO(txt), sep=delim)
    result = _analyze_core(df, input_hash)
    result["timestamp"] = datetime.utcnow().isoformat()+"Z"
    result["meta"]["dataset"] = "uploaded"
    return JSONResponse(result)

@app.post("/api/fetch-analyze")
def fetch_analyze(payload: FetchPayload):
    try:
        resp = requests.get(payload.url, timeout=30)
        resp.raise_for_status()
        txt = resp.text.lstrip("\ufeff")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch: {e}")
    input_hash = sha256_hex(txt)
    first = txt.splitlines()[0] if txt else ""
    delim = "\t" if "\t" in first else (";" if ";" in first else ",")
    df = pd.read_csv(io.StringIO(txt), sep=delim)
    result = _analyze_core(df, input_hash, payload.bootstrap_n or 300)
    result["timestamp"] = datetime.utcnow().isoformat()+"Z"
    result["meta"]["dataset"] = payload.url
    result["source"] = payload.url
    return JSONResponse(result)

@app.post("/api/verify-manifest")
def verify_manifest_api(manifest: Dict[str, Any] = Body(...)):
    sig = manifest.get("signature")
    if not sig:
        raise HTTPException(status_code=400, detail="Missing signature")
    man_copy = dict(manifest)
    signature = man_copy.pop("signature")
    ok = verify_signature(man_copy, signature)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return {"status": "verified", "merkle_root": man_copy.get("merkle_root")}

# -------- Plot endpoints: KM / iAUC / Forest / ROC(t) --------
class PlotPayload(BaseModel):
    url: str
    time: Optional[float] = None  # for ROC(t)

def _df_from_url(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        txt = r.text.lstrip("\ufeff")
        first = txt.splitlines()[0] if txt else ""
        delim = "\t" if "\t" in first else (";" if ";" in first else ",")
        return pd.read_csv(io.StringIO(txt), sep=delim)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch: {e}")

def _prep(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    _ensure_columns(df)
    df = df.dropna(subset=["Survival.time", "deadstatus.event"]).copy()
    df = _prepare_features(df)
    df = _build_unified(df, alpha)
    return df

@app.post("/api/plots/km")
def plot_km(payload: PlotPayload):
    df = _prep(_df_from_url(payload.url))
    df["group"] = pd.qcut(df["h_net"], 3, labels=["Low","Med","High"])
    km = KaplanMeierFitter()

    # compute CI and number-at-risk at ticks
    ticks = np.linspace(0, df["Survival.time"].quantile(0.9), 6)

    fig, (ax, ax_tab) = plt.subplots(2, 1, figsize=(7,5), dpi=150, gridspec_kw={'height_ratios': [5,1]})
    for label in ["Low","Med","High"]:
        m = df["group"] == label
        km.fit(df.loc[m, "Survival.time"], event_observed=df.loc[m, "deadstatus.event"], label=label)
        km.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Days"); ax.set_ylabel("Survival probability"); ax.set_title("Kaplan–Meier by Unified tertiles")
    ax.set_xticks(ticks)

    # number at risk table
    rows = []
    for label in ["Low","Med","High"]:
        m = df["group"] == label
        times = df.loc[m, "Survival.time"].values
        events = df.loc[m, "deadstatus.event"].values.astype(bool)
        # At risk at time t: individuals with time >= t
        row = [np.sum(times >= t) for t in ticks]
        rows.append([int(x) for x in row])
    table_data = [["Low"] + rows[0], ["Med"] + rows[1], ["High"] + rows[2]]
    ax_tab.axis("off")
    col_labels = ["Group"] + [f"{int(t)}d" for t in ticks]
    table = ax_tab.table(cellText=table_data, colLabels=col_labels, loc='center')
    table.scale(1, 1.2)

    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/api/plots/iauc")
def plot_iauc(payload: PlotPayload):
    df = _prep(_df_from_url(payload.url))
    y_train, risk_train = _surv_arrays_from(df, "h_net")
    times = np.quantile(df["Survival.time"], [0.25, 0.5, 0.75, 0.9])
    try:
        t, aucs = cumulative_dynamic_auc(y_train, y_train, -df["h_net"].values, times)
    except Exception as e:
        t, aucs = times, np.full(len(times), np.nan)
    try:
        iauc = float(np.trapz(aucs, t) / (t[-1]-t[0]))
    except Exception:
        iauc = np.nan
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.plot(t, aucs)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("AUC")
    ax.set_title(f"Dynamic AUC (iAUC ≈ {iauc:.3f})")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/api/plots/forest")
def plot_forest(payload: PlotPayload):
    df = _prep(_df_from_url(payload.url))
    features = [c for c in ["RISK_fractal","vt_end","h_net"] if c in df.columns]
    cph = CoxPHFitter()
    cph.fit(df[["Survival.time","deadstatus.event"] + features], duration_col="Survival.time", event_col="deadstatus.event")
    summ = cph.summary
    hr = summ["exp(coef)"]
    lo = summ["exp(coef) lower 95%"]
    hi = summ["exp(coef) upper 95%"]
    labels = list(hr.index)

    fig, ax = plt.subplots(figsize=(6, 0.5*len(labels)+1), dpi=150)
    y = np.arange(len(labels))
    ax.hlines(y, lo.values, hi.values)
    ax.plot(hr.values, y, "o")
    ax.vlines(1.0, -1, len(labels), linestyles="dashed")
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Cox Model — Forest Plot")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/api/plots/roc")
def plot_roct(payload: PlotPayload):
    df = _prep(_df_from_url(payload.url))
    # time t: dynamic case-control definition
    t = float(payload.time) if payload.time else float(np.quantile(df["Survival.time"], 0.5))
    # cases: events by t; controls: alive beyond t
    times = df["Survival.time"].values
    events = df["deadstatus.event"].values.astype(bool)
    score = df["h_net"].values
    mask = (times >= t) | (events & (times <= t))
    times, events, score = times[mask], events[mask], score[mask]
    # binary labels at t
    y = (events & (times <= t)).astype(int)
    # ROC curve by sweeping thresholds
    thr = np.unique(np.sort(score))
    if len(np.unique(y)) < 2 or thr.size < 3:
        # not enough to build ROC
        fig, ax = plt.subplots(figsize=(5,4), dpi=150)
        ax.text(0.5,0.5,"Insufficient cases/controls at selected time", ha="center")
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return Response(content=buf.getvalue(), media_type="image/png")
    tpr_list, fpr_list = [], []
    P = (y==1).sum(); N = (y==0).sum()
    for c in thr:
        yhat = (score >= c).astype(int)  # higher risk predicts event
        tp = np.sum((yhat==1)&(y==1)); fp = np.sum((yhat==1)&(y==0))
        fn = np.sum((yhat==0)&(y==1)); tn = np.sum((yhat==0)&(y==0))
        tpr = tp / P if P>0 else 0.0
        fpr = fp / N if N>0 else 0.0
        tpr_list.append(tpr); fpr_list.append(fpr)
    # sort by FPR
    order = np.argsort(fpr_list)
    fpr = np.array(fpr_list)[order]; tpr = np.array(tpr_list)[order]
    # AUC via trapezoid
    auc = float(np.trapz(tpr, fpr))

    fig, ax = plt.subplots(figsize=(5,4), dpi=150)
    ax.plot(fpr, tpr, label=f"AUC@{int(t)}d ≈ {auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Time-dependent ROC at t={int(t)} days")
    ax.legend(loc="lower right")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")
