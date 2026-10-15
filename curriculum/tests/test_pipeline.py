"""
Pipeline and Data Engineering Tests
Verifies lakehouse data integrity, schema consistency, and feature distributions.
"""

import os
import pytest
import pandas as pd


def test_processed_features_file_exists():
    parquet_path = "data/processed/student_features.parquet"
    csv_path = "data/processed/student_features.csv"
    assert os.path.exists(parquet_path) or os.path.exists(csv_path), "Master student features dataset must exist"


def test_student_features_schema_and_counts():
    parquet_path = "data/processed/student_features.parquet"
    df = pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else pd.read_csv("data/processed/student_features.csv")

    assert len(df) == 32593, f"Expected 32,593 students, got {len(df)}"
    
    # Check essential columns
    required_cols = [
        "code_module", "code_presentation", "id_student", "final_result",
        "final_result_binary", "is_at_risk", "total_vle_clicks", "vle_active_days",
        "avg_assessment_score", "assessment_completion_rate", "late_submission_rate"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check target binary integrity
    assert set(df["final_result_binary"].unique()).issubset({0, 1})
    assert set(df["is_at_risk"].unique()).issubset({0, 1})


def test_assessment_analytics_items():
    item_path = "data/processed/assessment_analytics.parquet"
    df = pd.read_parquet(item_path) if os.path.exists(item_path) else pd.read_csv("data/processed/assessment_analytics.csv")

    assert len(df) > 100, "Expected assessment items"
    assert "difficulty_index" in df.columns
    assert "pass_rate" in df.columns
    assert "late_submission_rate" in df.columns
