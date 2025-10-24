from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple
import io
import pandas as pd
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
except Exception:
    roc_auc_score = None
    brier_score_loss = None
    confusion_matrix = None

app = FastAPI(title="Fractal-UTL Demo API", version="1.0.0")

# CORS za Vite (localhost:5173). Dodaj po potrebi još origin-a.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UnifiedResult(BaseModel):
    cindex: float
    sens: Optional[float] = None
    spec: Optional[float] = None
    brier: Optional[float] = None
    hr: Optional[float] = None
    ci: Optional[Tuple[float, float]] = None
    notes: Optional[str] = None

@app.get("/api/ping")
def ping():
    # Ovdje ubaci realan ključ ako ga imaš
    return {"public_key": "UTL-DEMO-PUBKEY-123"}

@app.post("/api/compare", response_model=UnifiedResult)
async def compare(file: UploadFile = File(None)):
    """
    Prihvaća CSV/TSV s kolonama:
      - risk  (predviđena vjerojatnost [0,1])
      - event (ishod 0/1)
    Ako kolone ne postoje -> vrati DEMO rezultat.
    """
    if file is not None and file.filename:
        try:
            content = await file.read()
            # detektiraj delimiter
            text = content.decode("utf-8", errors="ignore")
            sep = "\t" if "\t" in text.splitlines()[0] else ","
            df = pd.read_csv(io.StringIO(text), sep=sep)

            # normaliziraj imena kolona
            cols = {c.lower().strip(): c for c in df.columns}
            has_risk = "risk" in cols
            has_event = "event" in cols

            if has_risk and has_event and roc_auc_score is not None:
                r = df[cols["risk"]].astype(float).clip(0, 1)
                y = df[cols["event"]].astype(int).clip(0, 1)

                # AUC kao “c-index” proxy za binarni ishod
                auc = float(roc_auc_score(y, r))

                # Brier
                brier = float(brier_score_loss(y, r))

                # Prag 0.5 sens/spec
                yhat = (r >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
                sens = float(tp / (tp + fn)) if (tp + fn) else None
                spec = float(tn / (tn + fp)) if (tn + fp) else None

                return UnifiedResult(
                    cindex=round(auc, 3),
                    sens=round(sens, 3) if sens is not None else None,
                    spec=round(spec, 3) if spec is not None else None,
                    brier=round(brier, 3),
                    hr=None,
                    ci=None,
                    notes="Computed from uploaded CSV.",
                )
        except Exception as e:
            # Fallback na demo
            return UnifiedResult(
                cindex=0.704, sens=0.78, spec=0.71, brier=0.186, hr=2.41, ci=(1.52, 3.82),
                notes=f"Demo result (parse error: {str(e)})"
            )

    # DEMO fallback (nema fajla ili nema kolona)
    return UnifiedResult(
        cindex=0.704, sens=0.78, spec=0.71, brier=0.186, hr=2.41, ci=(1.52, 3.82),
        notes="Demo result (no CSV or missing columns: need 'risk' and 'event')."
        @app.post("/api/fetch-analyze", response_model=UnifiedResult)
async def fetch_and_analyze(request: dict):
    """Fetch CSV from URL and analyze"""
    url = request.get("url")
    if not url:
        return UnifiedResult(
            cindex=0.0, sens=0.0, spec=0.0, brier=0.0, hr=0.0, ci=(0.0, 0.0),
            notes="Error: No URL provided"
        )
    
    try:
        import requests
        from io import StringIO
        
        # Fetch CSV from URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Read into pandas
        content = response.text
        df = pd.read_csv(StringIO(content), sep="\t" if "\t" in content.split("\n")[0] else ",")
        
        # Normalize column names
        cols = [c.lower().strip() for c in df.columns]
        has_risk = "risk" in cols
        has_event = "event" in cols
        
        if has_risk and has_event and roc_auc_score is not None:
            y = df[cols[cols.index("risk")]].astype(float).clip(0, 1)
            e = df[cols[cols.index("event")]].astype(int).clip(0, 1)
            
            # AUC as C-index proxy
            auc = float(roc_auc_score(e, y))
            
            # Brier
            brier = float(brier_score_loss(e, y))
            
            # Sens/spec at 0.5
            th, fn, fp, tn, tp = 0.5, 0, 0, 0, 0
            cm = confusion_matrix(e, (y >= th).astype(int))
            tn, fp, fn, tp = cm.ravel()
            sens = float(tp / (tp + fn)) if (tn + fp) else None
            spec = float(tn / (tn + fp)) if (tn + fp) else None
            
            return UnifiedResult(
                cindex=round(auc, 3),
                sens=round(sens, 3) if sens is not None else None,
                spec=round(spec, 3) if spec is not None else None,
                brier=round(brier, 3),
                hr=None,
                ci=None,
                notes=f"Analyzed from URL: {url}"
            )
    
    except Exception as e:
        return UnifiedResult(
            cindex=0.0, sens=0.0, spec=0.0, brier=0.0, hr=0.0, ci=(0.0, 0.0),
            notes=f"Error fetching/analyzing URL: {str(e)}"
        )
    
    # Fallback
    return UnifiedResult(
        cindex=0.704, sens=0.78, spec=0.71, brier=0.186, hr=2.41, ci=(1.52, 3.82),
        notes="Demo result (URL fetch: missing columns)"
    )
    )
