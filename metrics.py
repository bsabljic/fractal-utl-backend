"""
Statistical metrics for survival analysis - Windows compatible version.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix


def calculate_cindex_manual(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Manual C-index calculation (Harrell's concordance)."""
    try:
        n = len(time)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            if event[i] == 0:
                continue
            for j in range(n):
                if time[j] > time[i]:
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] < risk[j]:
                        discordant += 1
        
        total = concordant + discordant
        return concordant / total if total > 0 else 0.5
    except:
        return 0.5


def calculate_auc_roc(risk: np.ndarray, event: np.ndarray) -> float:
    try:
        if len(np.unique(event)) < 2:
            return 0.5
        return float(roc_auc_score(event, risk))
    except:
        return 0.5


def calculate_brier_score(risk: np.ndarray, event: np.ndarray) -> float:
    try:
        return float(brier_score_loss(event, risk))
    except:
        return 0.25


def calculate_sensitivity_specificity(
    risk: np.ndarray, 
    event: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[float, float]:
    try:
        predictions = (risk >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(event, predictions).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return float(sensitivity), float(specificity)
    except:
        return 0.5, 0.5


def calculate_logrank_test(
    time: np.ndarray, 
    event: np.ndarray, 
    risk: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float]:
    try:
        high_risk = risk >= threshold
        low_risk = risk < threshold
        
        if high_risk.sum() == 0 or low_risk.sum() == 0:
            return 0.0, 1.0
        
        result = logrank_test(
            time[high_risk], 
            time[low_risk],
            event[high_risk], 
            event[low_risk]
        )
        
        return float(result.test_statistic), float(result.p_value)
    except:
        return 0.0, 1.0


def calculate_cox_hazard_ratio(
    time: np.ndarray, 
    event: np.ndarray, 
    risk: np.ndarray
) -> Optional[float]:
    try:
        df = pd.DataFrame({
            'time': time,
            'event': event,
            'risk': risk
        })
        
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='event')
        
        hazard_ratio = float(np.exp(cph.params_['risk']))
        return hazard_ratio
    except:
        return None


def analyze_binary_classification(risk: np.ndarray, event: np.ndarray) -> Dict:
    auc = calculate_auc_roc(risk, event)
    brier = calculate_brier_score(risk, event)
    sensitivity, specificity = calculate_sensitivity_specificity(risk, event)
    
    return {
        "auc_roc": round(auc, 3),
        "brier_score": round(brier, 3),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
    }


def analyze_survival_data(
    time: np.ndarray, 
    event: np.ndarray, 
    risk: Optional[np.ndarray] = None
) -> Dict:
    metrics = {}
    
    if risk is not None:
        cindex = calculate_cindex_manual(risk, time, event)
        metrics["cindex"] = round(cindex, 3)
        
        test_stat, p_value = calculate_logrank_test(time, event, risk)
        metrics["logrank_test_statistic"] = round(test_stat, 3)
        metrics["logrank_lgroup_p"] = round(p_value, 4)
        
        hazard_ratio = calculate_cox_hazard_ratio(time, event, risk)
        if hazard_ratio is not None:
            metrics["hazard_ratio"] = round(hazard_ratio, 3)
        
        binary_metrics = analyze_binary_classification(risk, event)
        metrics.update(binary_metrics)
    else:
        n_events = int(event.sum())
        n_censored = int((1 - event).sum())
        median_followup = float(np.median(time))
        
        metrics["n_events"] = n_events
        metrics["n_censored"] = n_censored
        metrics["median_followup_time"] = round(median_followup, 2)
    
    return metrics
