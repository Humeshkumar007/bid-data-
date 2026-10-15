"""
Curriculum Effectiveness Analysis - FastAPI Production Backend
Exposes analytics endpoints, model evaluation, SHAP explainability,
real-time prediction simulation, and at-risk student monitoring.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.api.schemas import StudentPredictionRequest, PredictionResponse
from src.analytics.curriculum_analytics import get_course_performance_summary, get_demographic_performance
from src.analytics.assessment_analytics import get_assessment_effectiveness_summary
from src.analytics.engagement_analytics import get_engagement_summary
from src.analytics.statistical_tests import run_statistical_analysis
from src.ml.explainability import ModelExplainer

app = FastAPI(
    title="Curriculum Effectiveness Analysis API",
    description="Big Data & Learning Analytics Platform based on OULAD dataset",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize explainability model
try:
    explainer = ModelExplainer()
except Exception as e:
    print(f"[API Warning] Explainer initialization deferred: {e}")
    explainer = None


@app.get("/api/health")
def health_check():
    features_exist = os.path.exists("data/processed/student_features.parquet")
    models_exist = os.path.exists("data/models/best_model.joblib")
    return {
        "status": "healthy",
        "dataset_ready": features_exist,
        "models_ready": models_exist,
        "engine": "PySpark / Vectorized Lakehouse",
        "version": "1.0.0"
    }


@app.get("/api/curriculum/performance")
def curriculum_performance():
    """Returns course and presentation level performance summaries and curriculum review alerts."""
    try:
        return get_course_performance_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curriculum/demographics")
def demographic_performance():
    """Returns academic outcome distributions across demographic factors."""
    try:
        return get_demographic_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curriculum/assessments")
def assessment_effectiveness():
    """Returns TMA vs CMA vs Exam metrics, difficulty index, discrimination, and bottleneck assessments."""
    try:
        return get_assessment_effectiveness_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curriculum/engagement")
def student_engagement():
    """Returns VLE click dynamics, activity type impacts, and temporal engagement trends."""
    try:
        return get_engagement_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curriculum/statistical")
def statistical_tests():
    """Returns Pearson/Spearman correlation matrices, OLS regression, ANOVA tests, and findings."""
    try:
        return run_statistical_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/evaluation")
def model_evaluation():
    """Returns performance comparisons for Logistic Regression, Decision Tree, Random Forest, Gradient Boosting."""
    metrics_path = "data/models/model_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics not found. Run model training first.")
    with open(metrics_path, "r") as f:
        return json.load(f)


@app.get("/api/model/explainability/global")
def global_explainability():
    """Returns global feature importances and key factor rankings."""
    global explainer
    if explainer is None:
        explainer = ModelExplainer()
    return explainer.get_global_explanations()


@app.post("/api/model/predict", response_model=PredictionResponse)
def predict_student_outcome(student: StudentPredictionRequest):
    """
    Live real-time student outcome predictor.
    Returns:
      - Predicted outcome (Pass vs Fail)
      - Probability of Passing & Failing
      - Risk Tier (High, Medium, Low)
      - Local SHAP waterfall feature attribution breakdown
      - Actionable Pedagogical Interventions
    """
    global explainer
    if explainer is None:
        explainer = ModelExplainer()
    student_dict = student.model_dump()
    return explainer.explain_individual(student_dict)


@app.get("/api/students/at-risk")
def get_at_risk_students(
    module: Optional[str] = Query(None, description="Filter by course module"),
    limit: int = Query(25, ge=5, le=100, description="Max rows to return")
):
    """Returns filtered real student records identified as high or medium risk for proactive intervention."""
    data_path = "data/processed/student_features.parquet"
    if not os.path.exists(data_path):
        data_path = "data/processed/student_features.csv"
    
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    
    # Filter to at-risk students (withdrawn or failed, or low active days)
    at_risk_df = df[df["is_at_risk"] == 1].copy()
    if module:
        at_risk_df = at_risk_df[at_risk_df["code_module"] == module]

    # Sort by active days ascending (most disengaged first)
    at_risk_df = at_risk_df.sort_values(by=["vle_active_days", "avg_assessment_score"], ascending=[True, True])

    sample = at_risk_df.head(limit)[[
        "id_student", "code_module", "code_presentation", "gender", "age_band",
        "highest_education", "studied_credits", "total_vle_clicks", "vle_active_days",
        "avg_assessment_score", "late_submission_rate", "final_result"
    ]].to_dict(orient="records")

    return {
        "total_at_risk_count": int(len(at_risk_df)),
        "students": sample
    }


# Mount web application static assets
web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(web_dir, "index.html"))
