"""
Student Engagement Analytics Module
Analyzes VLE activity patterns, temporal click distributions,
active study days, resource category impact, and relationship with academic performance.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def load_student_data() -> pd.DataFrame:
    parquet_path = "data/processed/student_features.parquet"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.read_csv("data/processed/student_features.csv")


def get_engagement_summary() -> Dict[str, Any]:
    df = load_student_data()

    # 1. Engagement by Final Academic Outcome (Pass, Distinction, Fail, Withdrawn)
    outcome_engagement = df.groupby("final_result").agg(
        student_count=("id_student", "count"),
        avg_total_clicks=("total_vle_clicks", lambda s: round(s.mean(), 1)),
        median_total_clicks=("total_vle_clicks", lambda s: round(s.median(), 1)),
        avg_active_days=("vle_active_days", lambda s: round(s.mean(), 1)),
        avg_pre_clicks=("pre_course_clicks", lambda s: round(s.mean(), 1)),
        avg_early_clicks=("early_course_clicks", lambda s: round(s.mean(), 1)),
        avg_mid_clicks=("mid_course_clicks", lambda s: round(s.mean(), 1)),
        avg_late_clicks=("late_course_clicks", lambda s: round(s.mean(), 1)),
        avg_assessment_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2))
    ).loc[["Distinction", "Pass", "Fail", "Withdrawn"]].reset_index().to_dict(orient="records")

    # 2. Activity Type Click Breakdown across outcomes
    activity_cols = ["click_forum", "click_oucontent", "click_quiz", "click_resource", "click_subpage", "click_url", "click_homepage"]
    activity_breakdown = {}
    for col in activity_cols:
        clean_name = col.replace("click_", "")
        activity_breakdown[clean_name] = {
            "overall_avg": round(float(df[col].mean()), 1),
            "pass_avg": round(float(df[df["final_result_binary"] == 1][col].mean()), 1),
            "fail_avg": round(float(df[df["final_result_binary"] == 0][col].mean()), 1),
            "correlation_with_pass": round(float(df[col].corr(df["final_result_binary"])), 3)
        }

    # 3. Active Days Tiers vs Passing Probability
    day_bins = [-1, 0, 10, 30, 60, 100, 500]
    day_labels = ["0 Days", "1-10 Days", "11-30 Days", "31-60 Days", "61-100 Days", ">100 Days"]
    df["active_days_tier"] = pd.cut(df["vle_active_days"], bins=day_bins, labels=day_labels)
    
    tier_engagement = df.groupby("active_days_tier", observed=False).agg(
        students=("id_student", "count"),
        pass_rate=("final_result_binary", lambda s: round(s.mean() * 100, 2)),
        avg_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2)),
        avg_clicks=("total_vle_clicks", lambda s: round(s.mean(), 1))
    ).reset_index()
    tier_engagement["active_days_tier"] = tier_engagement["active_days_tier"].astype(str)

    # 4. Pre-course Engagement Impact (Early onboarding effect)
    df["has_pre_course_activity"] = (df["pre_course_clicks"] > 0).astype(int)
    pre_course_impact = df.groupby("has_pre_course_activity").agg(
        students=("id_student", "count"),
        pass_rate=("final_result_binary", lambda s: round(s.mean() * 100, 2)),
        withdrawal_rate=("final_result", lambda s: round((s == "Withdrawn").mean() * 100, 2)),
        avg_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2))
    ).reset_index().to_dict(orient="records")

    # 5. Module Engagement Comparison
    module_engagement = df.groupby("code_module").agg(
        avg_clicks=("total_vle_clicks", lambda s: round(s.mean(), 1)),
        avg_active_days=("vle_active_days", lambda s: round(s.mean(), 1)),
        avg_forum_clicks=("click_forum", lambda s: round(s.mean(), 1)),
        avg_quiz_clicks=("click_quiz", lambda s: round(s.mean(), 1)),
        avg_content_clicks=("click_oucontent", lambda s: round(s.mean(), 1))
    ).reset_index().to_dict(orient="records")

    return {
        "outcome_engagement": outcome_engagement,
        "activity_breakdown": activity_breakdown,
        "active_days_tiers": tier_engagement.to_dict(orient="records"),
        "pre_course_impact": pre_course_impact,
        "module_engagement": module_engagement
    }
