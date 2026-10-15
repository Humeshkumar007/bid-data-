"""
Curriculum Analytics Module
Computes course-level, module-level, and presentation-level performance metrics,
pass/fail rates, withdrawal rates, and curriculum benchmark comparisons.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def load_dataset(data_path: str = "data/processed/student_features.parquet") -> pd.DataFrame:
    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    csv_path = data_path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Processed dataset not found at {data_path} or {csv_path}")


def get_course_performance_summary(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Computes overall summary KPIs and course-by-course metrics:
    - Enrollment counts
    - Pass rate, Distinction rate, Fail rate, Withdrawal rate
    - Average assessment score
    - Average VLE clicks
    - Course difficulty ranking
    """
    if df is None:
        df = load_dataset()

    total_students = int(len(df))
    pass_count = int(df["final_result"].isin(["Pass", "Distinction"]).sum())
    fail_count = int((df["final_result"] == "Fail").sum())
    withdrawn_count = int((df["final_result"] == "Withdrawn").sum())
    distinction_count = int((df["final_result"] == "Distinction").sum())

    overall_pass_rate = round((pass_count / total_students) * 100.0, 2)
    overall_withdrawal_rate = round((withdrawn_count / total_students) * 100.0, 2)
    overall_fail_rate = round((fail_count / total_students) * 100.0, 2)
    overall_distinction_rate = round((distinction_count / total_students) * 100.0, 2)

    avg_score = round(float(df[df["avg_assessment_score"] > 0]["avg_assessment_score"].mean()), 2)
    avg_clicks = round(float(df["total_vle_clicks"].mean()), 1)
    avg_active_days = round(float(df["vle_active_days"].mean()), 1)

    # Group by code_module
    module_groups = df.groupby("code_module").agg(
        enrolled=("id_student", "count"),
        pass_rate=("final_result", lambda s: round(s.isin(["Pass", "Distinction"]).mean() * 100, 2)),
        distinction_rate=("final_result", lambda s: round((s == "Distinction").mean() * 100, 2)),
        fail_rate=("final_result", lambda s: round((s == "Fail").mean() * 100, 2)),
        withdrawal_rate=("final_result", lambda s: round((s == "Withdrawn").mean() * 100, 2)),
        avg_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2)),
        avg_clicks=("total_vle_clicks", lambda s: round(s.mean(), 1)),
        avg_active_days=("vle_active_days", lambda s: round(s.mean(), 1)),
        late_sub_rate=("late_submission_rate", lambda s: round(s.mean() * 100, 2)),
        assessment_completion=("assessment_completion_rate", lambda s: round(s.mean() * 100, 2))
    ).reset_index()

    # Presentation breakdown
    presentation_groups = df.groupby(["code_module", "code_presentation"]).agg(
        enrolled=("id_student", "count"),
        pass_rate=("final_result", lambda s: round(s.isin(["Pass", "Distinction"]).mean() * 100, 2)),
        withdrawal_rate=("final_result", lambda s: round((s == "Withdrawn").mean() * 100, 2)),
        avg_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2)),
        avg_clicks=("total_vle_clicks", lambda s: round(s.mean(), 1))
    ).reset_index()

    # Curriculum recommendations & bottleneck detection
    underperforming_courses = module_groups[module_groups["pass_rate"] < overall_pass_rate][
        ["code_module", "pass_rate", "withdrawal_rate", "avg_score"]
    ].to_dict(orient="records")

    high_withdrawal_courses = module_groups[module_groups["withdrawal_rate"] > overall_withdrawal_rate][
        ["code_module", "withdrawal_rate", "pass_rate"]
    ].to_dict(orient="records")

    return {
        "overall_kpis": {
            "total_enrollments": total_students,
            "overall_pass_rate": overall_pass_rate,
            "overall_withdrawal_rate": overall_withdrawal_rate,
            "overall_fail_rate": overall_fail_rate,
            "overall_distinction_rate": overall_distinction_rate,
            "avg_assessment_score": avg_score,
            "avg_vle_clicks": avg_clicks,
            "avg_active_days": avg_active_days,
            "active_courses_count": int(df["code_module"].nunique())
        },
        "modules": module_groups.to_dict(orient="records"),
        "presentations": presentation_groups.to_dict(orient="records"),
        "curriculum_review_alerts": {
            "underperforming_modules": underperforming_courses,
            "high_withdrawal_modules": high_withdrawal_courses
        }
    }


def get_demographic_performance(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Breakdown of educational outcomes by demographic indicators."""
    if df is None:
        df = load_dataset()

    def get_breakdown(column: str) -> List[Dict[str, Any]]:
        return df.groupby(column).agg(
            count=("id_student", "count"),
            pass_rate=("final_result", lambda s: round(s.isin(["Pass", "Distinction"]).mean() * 100, 2)),
            withdrawal_rate=("final_result", lambda s: round((s == "Withdrawn").mean() * 100, 2)),
            avg_score=("avg_assessment_score", lambda s: round(s[s > 0].mean(), 2)),
            avg_clicks=("total_vle_clicks", lambda s: round(s.mean(), 1))
        ).reset_index().to_dict(orient="records")

    return {
        "highest_education": get_breakdown("highest_education"),
        "age_band": get_breakdown("age_band"),
        "gender": get_breakdown("gender"),
        "imd_band": get_breakdown("imd_band"),
        "disability": get_breakdown("disability")
    }
