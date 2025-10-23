from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import Dict, List, Optional
import numpy as np
from metrics import analyze_survival_data, analyze_binary_classification

app = FastAPI(title="Fractal-UTL API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RISK_ALIASES = ['risk', 'risk_score', 'prediction', 'prob', 'probability', 'score']
EVENT_ALIASES = ['event', 'status', 'outcome', 'death', 'recurrence', 'relapse']
TIME_ALIASES = ['time', 'survival_time', 'followup', 'follow_up', 'duration', 'days', 'months']

def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    df_cols_lower = {col.lower(): col for col in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    return None

def validate_and_parse_csv(file_content: bytes) -> Dict:
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    
    risk_col = find_column(df, RISK_ALIASES)
    event_col = find_column(df, EVENT_ALIASES)
    time_col = find_column(df, TIME_ALIASES)
    
    if not event_col:
        raise HTTPException(
            status_code=400, 
            detail=f"Event column not found. Expected one of: {EVENT_ALIASES}"
        )
    
    if not time_col:
        raise HTTPException(
            status_code=400, 
            detail=f"Time column not found. Expected one of: {TIME_ALIASES}"
        )
    
    event = df[event_col].values
    time = df[time_col].values
    risk = df[risk_col].values if risk_col else None
    
    if not np.issubdtype(event.dtype, np.number):
        raise HTTPException(status_code=400, detail="Event column must be numeric (0/1)")
    
    if not np.issubdtype(time.dtype, np.number):
        raise HTTPException(status_code=400, detail="Time column must be numeric")
    
    if risk is not None and not np.issubdtype(risk.dtype, np.number):
        raise HTTPException(status_code=400, detail="Risk column must be numeric")
    
    return {
        "risk": risk,
        "event": event,
        "time": time,
        "n_samples": len(df),
        "has_risk": risk is not None
    }

@app.get("/api/ping")
def ping():
    return {"status": "ok", "message": "Fractal-UTL API is running"}

@app.post("/api/compare")
async def compare_models(file: UploadFile = File(...)):
    content = await file.read()
    data = validate_and_parse_csv(content)
    
    if data["has_risk"]:
        # Calculate all metrics
        survival_metrics = analyze_survival_data(
            time=data["time"],
            event=data["event"],
            risk=data["risk"]
        )
        
        # Calculate confidence intervals for C-index (approximate)
        n = data["n_samples"]
        cindex = survival_metrics.get("cindex", 0.5)
        se = np.sqrt(cindex * (1 - cindex) / n)  # Approximate SE
        ci_lower = max(0.0, cindex - 1.96 * se)
        ci_upper = min(1.0, cindex + 1.96 * se)
        
        # Calculate improvement over random (0.5)
        improvement = (cindex - 0.5) * 2  # Scale to [0, 1]
        
        # Format response to match frontend expectations
        response = {
            "success": True,
            "n_samples": data["n_samples"],
            "metrics": {
                "unified": {
                    "cindex": survival_metrics.get("cindex", 0.5),
                    "ci_lower": round(ci_lower, 3),
                    "ci_upper": round(ci_upper, 3),
                    "improvement": round(improvement, 3),
                    "logrank_3group_p": survival_metrics.get("logrank_lgroup_p", 1.0),
                    "hazard_ratio": survival_metrics.get("hazard_ratio", 1.0),
                    "auc_roc": survival_metrics.get("auc_roc", 0.5),
                    "brier_score": survival_metrics.get("brier_score", 0.25),
                    "sensitivity": survival_metrics.get("sensitivity", 0.5),
                    "specificity": survival_metrics.get("specificity", 0.5),
                }
            }
        }
    else:
        survival_metrics = analyze_survival_data(
            time=data["time"],
            event=data["event"],
            risk=None
        )
        
        response = {
            "success": True,
            "n_samples": data["n_samples"],
            "metrics": {
                "basic_stats": survival_metrics
            },
            "note": "No risk scores provided - only basic survival statistics calculated"
        }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
