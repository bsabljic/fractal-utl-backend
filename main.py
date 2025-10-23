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

# Alias mapping za kolone
RISK_ALIASES = ['risk', 'risk_score', 'prediction', 'prob', 'probability', 'score']
EVENT_ALIASES = ['event', 'status', 'outcome', 'death', 'recurrence', 'relapse']
TIME_ALIASES = ['time', 'survival_time', 'followup', 'follow_up', 'duration', 'days', 'months']

def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """Find column by checking aliases (case-insensitive)."""
    df_cols_lower = {col.lower(): col for col in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    return None

def validate_and_parse_csv(file_content: bytes) -> Dict:
    """Parse CSV and validate required columns."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    
    # Find columns by aliases
    risk_col = find_column(df, RISK_ALIASES)
    event_col = find_column(df, EVENT_ALIASES)
    time_col = find_column(df, TIME_ALIASES)
    
    # Validation
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
    
    # Extract data
    event = df[event_col].values
    time = df[time_col].values
    risk = df[risk_col].values if risk_col else None
    
    # Validate data types
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
    """Health check endpoint."""
    return {"status": "ok", "message": "Fractal-UTL API is running"}

@app.post("/api/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    Analyze survival data from uploaded CSV.
    
    Expected columns (case-insensitive):
    - risk/risk_score/prediction (optional)
    - event/status/outcome (required)
    - time/survival_time/followup (required)
    """
    
    # Read file
    content = await file.read()
    
    # Parse and validate
    data = validate_and_parse_csv(content)
    
    # Calculate metrics
    if data["has_risk"]:
        # Full analysis with risk scores
        survival_metrics = analyze_survival_data(
            time=data["time"],
            event=data["event"],
            risk=data["risk"]
        )
        
        response = {
            "success": True,
            "n_samples": data["n_samples"],
            "metrics": {
                "unified": survival_metrics
            }
        }
    else:
        # Basic survival stats without risk
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
