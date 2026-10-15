"""
Assessment Effectiveness Analytics Module
Analyzes TMA vs CMA vs Exam performance, difficulty vs weight alignment,
late submission impacts, score distributions, and bottleneck assessments.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def load_assessment_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat_path = "data/processed/student_features.parquet"
    item_path = "data/processed/assessment_analytics.parquet"
    
    df_students = pd.read_parquet(feat_path) if os.path.exists(feat_path) else pd.read_csv("data/processed/student_features.csv")
    df_items = pd.read_parquet(item_path) if os.path.exists(item_path) else pd.read_csv("data/processed/assessment_analytics.csv")
    
    return df_students, df_items


def get_assessment_effectiveness_summary() -> Dict[str, Any]:
    df_students, df_items = load_assessment_data()

    # 1. Comparison by Assessment Type (TMA vs CMA vs Exam)
    type_summary = df_items.groupby("assessment_type").agg(
        count=("id_assessment", "count"),
        avg_score=("avg_score", "mean"),
        pass_rate=("pass_rate", "mean"),
        avg_weight=("weight", "mean"),
        avg_late_rate=("late_submission_rate", "mean"),
        avg_submissions=("total_submissions", "mean")
    ).reset_index()

    type_summary = type_summary.fillna(0.0).round(2).to_dict(orient="records")

    # 2. Score distributions across types
    # Quantiles for TMA, CMA, Exam
    distributions = {}
    for atype in ["TMA", "CMA", "Exam"]:
        sub = df_items[df_items["assessment_type"] == atype]["avg_score"].dropna()
        if len(sub) > 0:
            distributions[atype] = {
                "mean": round(float(sub.mean()), 2),
                "median": round(float(sub.median()), 2),
                "min": round(float(sub.min()), 2),
                "max": round(float(sub.max()), 2),
                "std": round(float(sub.std()), 2)
            }

    # 3. Challenging / Bottleneck Assessments
    # Defined by: low pass rate (< 75%), or high difficulty (> 30), or high late rate (> 25%)
    challenging = df_items[
        (df_items["pass_rate"] < 75.0) | 
        (df_items["late_submission_rate"] > 25.0) | 
        (df_items["avg_score"] < 60.0)
    ].sort_values(by="pass_rate", ascending=True)

    challenging_records = challenging[[
        "id_assessment", "code_module", "code_presentation", "assessment_type",
        "weight", "avg_score", "pass_rate", "late_submission_rate", "difficulty_index"
    ]].head(15).fillna(0.0).round(2).to_dict(orient="records")

    # 4. Weight vs Difficulty Alignment Analysis
    # Identifying high-weight assessments with high failure risk (critical curriculum risk points)
    critical_risk_assessments = df_items[
        (df_items["weight"] >= 20) & (df_items["avg_score"] < 65)
    ].sort_values(by="weight", ascending=False).fillna(0.0).round(2).to_dict(orient="records")

    # 5. Scatter Plot points: weight vs avg_score vs pass_rate for all assessments
    scatter_data = df_items[[
        "id_assessment", "code_module", "code_presentation", "assessment_type",
        "weight", "avg_score", "pass_rate", "late_submission_rate", "difficulty_index"
    ]].dropna().fillna(0.0).round(2).to_dict(orient="records")

    # 6. Student Assessment Behavior Metrics (Late submission vs final outcome)
    late_vs_outcome = df_students.groupby(
        pd.cut(df_students["late_submission_rate"], bins=[-0.01, 0, 0.25, 0.5, 1.0], labels=["0% Late", "1-25% Late", "26-50% Late", ">50% Late"])
    ).agg(
        student_count=("id_student", "count"),
        pass_rate=("final_result_binary", lambda x: round(x.mean() * 100, 2)),
        avg_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2))
    ).reset_index()
    late_vs_outcome["late_submission_rate"] = late_vs_outcome["late_submission_rate"].astype(str)
    late_vs_outcome = late_vs_outcome.fillna(0.0)

    return {
        "assessment_type_summary": type_summary,
        "score_distributions": distributions,
        "challenging_assessments": challenging_records,
        "critical_weight_risks": critical_risk_assessments,
        "scatter_data": scatter_data,
        "late_submission_impact": late_vs_outcome.to_dict(orient="records")
    }
