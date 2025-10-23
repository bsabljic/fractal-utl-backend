from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import re
from io import StringIO
import pandas as pd

app = FastAPI(title="Fractal-UTL API")

# CORS middleware - dozvoli sve origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== HELPER FUNCTIONS =====

# Alias patterns za fleksibilno prepoznavanje kolona
RISK_ALIASES  = [r"^risk$", r"^risk_?score$", r"^unified(_risk)?$", r"^score$", r"^pred(iction)?$", r"^prob(a)?$", r"^p(_\w+)?$"]
EVENT_ALIASES = [r"^event$", r"^status$", r"^dead(status)?(\.event)?$", r"^death$", r"^label$", r"^outcome$", r"^y$", r"^event_observed$"]
TIME_ALIASES  = [r"^time$", r"^survival(\.time)?$", r"^os(_time)?$", r"^followup$", r"^t$"]

def _normalize_cols(df: pd.DataFrame):
    """Vraća mapu {original_name: normalized_lowercase}"""
    return {c: c.strip().lower() for c in df.columns}

def _find_col(norm_cols: dict, patterns):
    """Traži kolonu koja match-uje bilo koji pattern iz liste"""
    for orig, low in norm_cols.items():
        for pat in patterns:
            if re.match(pat, low):
                return orig
    return None

def infer_columns(df: pd.DataFrame):
    """Automatski detektuje risk/event/time kolone korištenjem aliasa"""
    norm = _normalize_cols(df)
    risk  = _find_col(norm, RISK_ALIASES)
    event = _find_col(norm, EVENT_ALIASES)
    time  = _find_col(norm, TIME_ALIASES)
    return {"risk": risk, "event": event, "time": time}

# ===== API ENDPOINTS =====

@app.get("/api/ping")
def ping():
    return {"status": "ok"}

@app.post("/api/compare")
async def compare(file: UploadFile = File(None), demo: str = None):
    """
    Analyze CSV/TSV with flexible column detection.
    Supports: risk+event (binary) or time+event+risk (survival)
    """
    
    # Demo mode: eksplicitno tražen
    if demo == "1":
        return JSONResponse(status_code=200, content={
            "public_key": "UTL-DEMO-PUBKEY-123",
            "meta": {"dataset": "Demo Dataset", "n_patients": 100},
            "metrics": {
                "unified": {
                    "cindex": 0.692,
                    "auc_roc": 0.692,
                    "brier_score": 0.18,
                    "sensitivity": 0.75,
                    "specificity": 0.68,
                    "logrank_lgroup_p": 0.042
                }
            }
        })
    
    # Provjera da li je file poslan
    if file is None or not getattr(file, "filename", ""):
        return JSONResponse(
            status_code=400,
            content={
                "error": "no_file",
                "message": "Nije poslana datoteka. Molimo upload-ujte CSV/TSV fajl sa kolonama: risk+event ili time+event+risk."
            }
        )
    
    try:
        # Čitanje i parsiranje
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        
        # Autodetekcija delimitera
        first_line = text.splitlines()[0] if text else ""
        delimiter = "," if "," in first_line else "\t" if "\t" in first_line else ";"
        
        # Parse CSV
        df = pd.read_csv(StringIO(text), sep=delimiter)
        
        if df.empty:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "empty_file",
                    "message": "Fajl je prazan ili nema validnih redova."
                }
            )
        
        # Automatsko prepoznavanje kolona
        cols = infer_columns(df)
        
        # Validacija - moramo imati barem event kolonu
        if cols["event"] is None:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "missing_columns",
                    "message": f"Nedostaje 'event' kolona. Pronađene kolone: {list(df.columns)}. Podržavamo aliase: {EVENT_ALIASES}"
                }
            )
        
        # Scenario 1: Binary classification (risk + event)
        if cols["risk"] is not None and cols["event"] is not None:
            risk_vals = df[cols["risk"]].dropna()
            event_vals = df[cols["event"]].dropna()
            
            # Validacija risk ∈ [0,1]
            if (risk_vals < 0).any() or (risk_vals > 1).any():
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "invalid_risk_range",
                        "message": f"'risk' vrijednosti moraju biti u rasponu [0,1]. Pronađen min={risk_vals.min():.3f}, max={risk_vals.max():.3f}"
                    }
                )
            
            # Mock metrike (u realnom scenariju ovdje bi bila statistička analiza)
            n_patients = len(df)
            n_events = int(event_vals.sum())
            
            return JSONResponse(status_code=200, content={
                "public_key": f"UTL-UPLOAD-{file.filename[:20]}",
                "meta": {
                    "dataset": file.filename,
                    "n_patients": n_patients,
                    "n_events": n_events,
                    "detected_columns": cols
                },
                "metrics": {
                    "unified": {
                        "cindex": 0.65,  # TODO: Računaj stvarnu C-index
                        "auc_roc": 0.68,
                        "brier_score": 0.21,
                        "sensitivity": 0.72,
                        "specificity": 0.64
                    }
                }
            })
        
        # Scenario 2: Survival analysis (time + event + optional risk)
        elif cols["time"] is not None and cols["event"] is not None:
            time_vals = df[cols["time"]].dropna()
            event_vals = df[cols["event"]].dropna()
            
            # Validacija time > 0
            if (time_vals <= 0).any():
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "invalid_time",
                        "message": f"'time' vrijednosti moraju biti > 0. Pronađen min={time_vals.min():.3f}"
                    }
                )
            
            n_patients = len(df)
            n_events = int(event_vals.sum())
            
            return JSONResponse(status_code=200, content={
                "public_key": f"UTL-SURVIVAL-{file.filename[:20]}",
                "meta": {
                    "dataset": file.filename,
                    "n_patients": n_patients,
                    "n_events": n_events,
                    "detected_columns": cols
                },
                "metrics": {
                    "unified": {
                        "cindex": 0.67,  # TODO: Računaj stvarnu C-index iz survival data
                        "logrank_lgroup_p": 0.038,
                        "hazard_ratio": 2.14
                    }
                }
            })
        
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "unsupported_format",
                    "message": f"Nisu pronađene validne kolone. Detektovano: {cols}. Podržavamo: (a) risk+event ili (b) time+event+risk."
                }
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "processing_error",
                "message": f"Greška pri obradi fajla: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
