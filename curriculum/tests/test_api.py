"""
FastAPI Endpoints Integration Tests
Tests all analytics, prediction, explainability, and at-risk monitoring endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dataset_ready"] is True
    assert data["models_ready"] is True


def test_curriculum_performance_endpoint():
    response = client.get("/api/curriculum/performance")
    assert response.status_code == 200
    data = response.json()
    assert "overall_kpis" in data
    assert "modules" in data
    assert len(data["modules"]) == 7
    assert data["overall_kpis"]["total_enrollments"] == 32593


def test_curriculum_assessments_endpoint():
    response = client.get("/api/curriculum/assessments")
    assert response.status_code == 200
    data = response.json()
    assert "assessment_type_summary" in data
    assert "challenging_assessments" in data
    assert "scatter_data" in data


def test_curriculum_engagement_endpoint():
    response = client.get("/api/curriculum/engagement")
    assert response.status_code == 200
    data = response.json()
    assert "outcome_engagement" in data
    assert "active_days_tiers" in data


def test_statistical_tests_endpoint():
    response = client.get("/api/curriculum/statistical")
    assert response.status_code == 200
    data = response.json()
    assert "correlation_matrix" in data
    assert "ols_regression" in data
    assert "curriculum_findings" in data


def test_model_evaluation_endpoint():
    response = client.get("/api/model/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "best_model_name" in data
    assert "models" in data
    assert "Random Forest" in data["models"]
    assert "Gradient Boosting" in data["models"]


def test_model_prediction_and_shap_endpoint():
    payload = {
        "code_module": "BBB",
        "gender": "F",
        "highest_education": "A Level or Equivalent",
        "age_band": "0-35",
        "disability": "N",
        "studied_credits": 60,
        "num_of_prev_attempts": 0,
        "days_before_course_registered": 30,
        "total_assessments_submitted": 4,
        "avg_assessment_score": 75.0,
        "weighted_score": 70.0,
        "late_submission_rate": 0.0,
        "avg_submission_delay": -2.0,
        "assessment_completion_rate": 0.8,
        "tma_submitted": 3,
        "tma_avg_score": 78.0,
        "cma_submitted": 1,
        "cma_avg_score": 70.0,
        "total_vle_clicks": 1500,
        "vle_active_days": 55,
        "clicks_per_active_day": 27.0,
        "pre_course_clicks": 120,
        "early_course_clicks": 450,
        "click_forum": 350,
        "click_oucontent": 650,
        "click_quiz": 150,
        "click_resource": 90,
        "click_url": 50
    }
    response = client.post("/api/model/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["outcome"] in ["Pass", "Fail"]
    assert "probability_pass" in data["prediction"]
    assert "top_contributing_factors" in data
    assert len(data["top_contributing_factors"]) > 0
    assert "recommended_interventions" in data


def test_at_risk_students_endpoint():
    response = client.get("/api/students/at-risk?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "students" in data
    assert len(data["students"]) <= 10
    assert data["total_at_risk_count"] > 0
