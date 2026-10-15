"""
Curriculum Effectiveness Analysis - High-Performance Vectorized ETL Pipeline
Processes raw OULAD relational datasets into Lakehouse Parquet analytical storage
and produces gold feature-engineered datasets for ML & curriculum analytics.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def ensure_directories():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("data/models", exist_ok=True)


def process_assessments(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes assessments.csv and studentAssessment.csv.
    Returns:
      - student_assessment_features: per (code_module, code_presentation, id_student)
      - assessment_item_analytics: per id_assessment (difficulty, discrimination, submission rates)
    """
    print("[ETL] Processing Assessments data...")
    assessments_path = os.path.join(raw_dir, "assessments.csv")
    student_assess_path = os.path.join(raw_dir, "studentAssessment.csv")

    assess_df = pd.read_csv(assessments_path)
    student_assess_df = pd.read_csv(student_assess_path)

    # Count total assessments planned per module presentation
    module_assess_counts = assess_df.groupby(["code_module", "code_presentation"])["id_assessment"].nunique().reset_index()
    module_assess_counts.rename(columns={"id_assessment": "total_module_assessments"}, inplace=True)

    # Merge assessment metadata
    merged = pd.merge(student_assess_df, assess_df, on="id_assessment", how="inner")

    # Calculate submission delay: positive = late, negative = early
    # Note: assessment date can be NaN for exams
    merged["submission_delay"] = merged["date_submitted"] - merged["date"].fillna(merged["date_submitted"])
    merged["is_late"] = (merged["submission_delay"] > 0).astype(int)
    merged["weighted_score_contrib"] = merged["score"] * (merged["weight"] / 100.0)

    # 1. Assessment Item Analytics (Level 1 Analysis from Slide 3 & 5)
    assess_item_metrics = merged.groupby("id_assessment").agg(
        total_submissions=("id_student", "count"),
        avg_score=("score", "mean"),
        median_score=("score", "median"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        pass_rate=("score", lambda s: (s >= 40.0).mean() * 100.0),
        high_score_rate=("score", lambda s: (s >= 70.0).mean() * 100.0),
        late_submission_rate=("is_late", lambda x: x.mean() * 100.0),
        avg_delay_days=("submission_delay", "mean")
    ).reset_index()

    assess_item_analytics = pd.merge(assess_df, assess_item_metrics, on="id_assessment", how="left")
    # Assessment difficulty index: 100 - avg_score (higher means more difficult)
    assess_item_analytics["difficulty_index"] = 100.0 - assess_item_analytics["avg_score"]
    
    # 2. Student-level assessment aggregations
    print("[ETL] Aggregating student-level assessment metrics...")
    
    # TMA, CMA, Exam splits
    tma_df = merged[merged["assessment_type"] == "TMA"].groupby(["code_module", "code_presentation", "id_student"]).agg(
        tma_submitted=("id_assessment", "count"),
        tma_avg_score=("score", "mean")
    ).reset_index()

    cma_df = merged[merged["assessment_type"] == "CMA"].groupby(["code_module", "code_presentation", "id_student"]).agg(
        cma_submitted=("id_assessment", "count"),
        cma_avg_score=("score", "mean")
    ).reset_index()

    exam_df = merged[merged["assessment_type"] == "Exam"].groupby(["code_module", "code_presentation", "id_student"]).agg(
        exam_submitted=("id_assessment", "count"),
        exam_score=("score", "mean")
    ).reset_index()

    # Overall student assessment metrics
    student_assess_agg = merged.groupby(["code_module", "code_presentation", "id_student"]).agg(
        total_assessments_submitted=("id_assessment", "count"),
        avg_assessment_score=("score", "mean"),
        median_assessment_score=("score", "median"),
        std_assessment_score=("score", "std"),
        min_assessment_score=("score", "min"),
        max_assessment_score=("score", "max"),
        weighted_score=("weighted_score_contrib", "sum"),
        total_late_submissions=("is_late", "sum"),
        late_submission_rate=("is_late", "mean"),
        avg_submission_delay=("submission_delay", "mean")
    ).reset_index()

    # Merge splits
    student_assess_features = pd.merge(student_assess_agg, module_assess_counts, on=["code_module", "code_presentation"], how="left")
    student_assess_features["assessment_completion_rate"] = (
        student_assess_features["total_assessments_submitted"] / student_assess_features["total_module_assessments"]
    ).clip(0.0, 1.0)

    student_assess_features = pd.merge(student_assess_features, tma_df, on=["code_module", "code_presentation", "id_student"], how="left")
    student_assess_features = pd.merge(student_assess_features, cma_df, on=["code_module", "code_presentation", "id_student"], how="left")
    student_assess_features = pd.merge(student_assess_features, exam_df, on=["code_module", "code_presentation", "id_student"], how="left")

    student_assess_features["std_assessment_score"] = student_assess_features["std_assessment_score"].fillna(0.0)
    student_assess_features["tma_submitted"] = student_assess_features["tma_submitted"].fillna(0)
    student_assess_features["cma_submitted"] = student_assess_features["cma_submitted"].fillna(0)
    student_assess_features["exam_submitted"] = student_assess_features["exam_submitted"].fillna(0)

    return student_assess_features, assess_item_analytics


def process_vle(raw_dir: str) -> pd.DataFrame:
    """
    Processes vle.csv and studentVle.csv (10.6M rows) in high-throughput chunks.
    Extracts total clicks, temporal trends (pre, early, mid, late), active days,
    and activity category engagement (forum, content, quiz, url, resource, etc.)
    """
    print("[ETL] Processing VLE interactions (10.6M rows)...")
    vle_path = os.path.join(raw_dir, "vle.csv")
    student_vle_path = os.path.join(raw_dir, "studentVle.csv")

    vle_df = pd.read_csv(vle_path)
    site_to_activity = dict(zip(vle_df["id_site"], vle_df["activity_type"]))

    chunk_size = 2_000_000
    student_clicks_chunks = []
    student_temporal_chunks = []
    student_activity_chunks = []
    student_active_days_chunks = []

    start_time = time.time()
    for idx, chunk in enumerate(pd.read_csv(student_vle_path, chunksize=chunk_size)):
        # 1. Total clicks
        agg_clicks = chunk.groupby(["code_module", "code_presentation", "id_student"], as_index=False)["sum_click"].sum()
        student_clicks_chunks.append(agg_clicks)

        # 2. Temporal engagement breakdown
        # date < 0: pre-course, 0-30: early weeks, 31-100: mid term, > 100: late term
        chunk["is_pre"] = (chunk["date"] < 0) * chunk["sum_click"]
        chunk["is_early"] = ((chunk["date"] >= 0) & (chunk["date"] <= 30)) * chunk["sum_click"]
        chunk["is_mid"] = ((chunk["date"] > 30) & (chunk["date"] <= 100)) * chunk["sum_click"]
        chunk["is_late_term"] = (chunk["date"] > 100) * chunk["sum_click"]

        agg_temporal = chunk.groupby(["code_module", "code_presentation", "id_student"], as_index=False)[
            ["is_pre", "is_early", "is_mid", "is_late_term"]
        ].sum()
        student_temporal_chunks.append(agg_temporal)

        # 3. Activity type mapping
        chunk["activity"] = chunk["id_site"].map(site_to_activity).fillna("other")
        chunk["click_forum"] = (chunk["activity"] == "forumng") * chunk["sum_click"]
        chunk["click_oucontent"] = (chunk["activity"] == "oucontent") * chunk["sum_click"]
        chunk["click_quiz"] = (chunk["activity"].isin(["quiz", "externalquiz"])) * chunk["sum_click"]
        chunk["click_resource"] = (chunk["activity"] == "resource") * chunk["sum_click"]
        chunk["click_subpage"] = (chunk["activity"] == "subpage") * chunk["sum_click"]
        chunk["click_url"] = (chunk["activity"] == "url") * chunk["sum_click"]
        chunk["click_homepage"] = (chunk["activity"] == "homepage") * chunk["sum_click"]

        agg_activity = chunk.groupby(["code_module", "code_presentation", "id_student"], as_index=False)[
            ["click_forum", "click_oucontent", "click_quiz", "click_resource", "click_subpage", "click_url", "click_homepage"]
        ].sum()
        student_activity_chunks.append(agg_activity)

        # 4. Active days (distinct dates)
        date_distinct = chunk[["code_module", "code_presentation", "id_student", "date"]].drop_duplicates()
        student_active_days_chunks.append(date_distinct)

        print(f"  - Chunk {idx+1} processed in {time.time() - start_time:.1f}s")

    print("[ETL] Consolidating VLE chunk aggregations...")
    # Final consolidations
    all_clicks = pd.concat(student_clicks_chunks).groupby(["code_module", "code_presentation", "id_student"], as_index=False)["sum_click"].sum()
    all_clicks.rename(columns={"sum_click": "total_vle_clicks"}, inplace=True)

    all_temporal = pd.concat(student_temporal_chunks).groupby(["code_module", "code_presentation", "id_student"], as_index=False).sum()
    all_temporal.rename(columns={
        "is_pre": "pre_course_clicks",
        "is_early": "early_course_clicks",
        "is_mid": "mid_course_clicks",
        "is_late_term": "late_course_clicks"
    }, inplace=True)

    all_activity = pd.concat(student_activity_chunks).groupby(["code_module", "code_presentation", "id_student"], as_index=False).sum()

    all_dates = pd.concat(student_active_days_chunks).drop_duplicates()
    active_days = all_dates.groupby(["code_module", "code_presentation", "id_student"], as_index=False)["date"].count()
    active_days.rename(columns={"date": "vle_active_days"}, inplace=True)

    # Merge all VLE student features
    vle_features = pd.merge(all_clicks, active_days, on=["code_module", "code_presentation", "id_student"], how="left")
    vle_features = pd.merge(vle_features, all_temporal, on=["code_module", "code_presentation", "id_student"], how="left")
    vle_features = pd.merge(vle_features, all_activity, on=["code_module", "code_presentation", "id_student"], how="left")

    vle_features["clicks_per_active_day"] = (
        vle_features["total_vle_clicks"] / vle_features["vle_active_days"].clip(lower=1)
    ).round(2)

    return vle_features


def build_analytical_dataset(raw_dir: str = RAW_DATA_DIR, output_dir: str = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """
    Executes the full pipeline:
    1. Loads studentInfo, studentRegistration, courses
    2. Merges assessment features
    3. Merges VLE engagement features
    4. Computes target labels (binary Pass/Fail and multi-class)
    5. Saves Lakehouse Parquet and CSV files
    """
    ensure_directories()
    start_total = time.time()

    print("[ETL] Loading Student Info, Registration, and Courses...")
    student_info = pd.read_csv(os.path.join(raw_dir, "studentInfo.csv"))
    student_reg = pd.read_csv(os.path.join(raw_dir, "studentRegistration.csv"))
    courses = pd.read_csv(os.path.join(raw_dir, "courses.csv"))

    # Registration features
    student_reg["days_before_course_registered"] = (-student_reg["date_registration"]).fillna(0)
    student_reg["has_unregistered"] = student_reg["date_unregistration"].notna().astype(int)
    student_reg["unregistration_day"] = student_reg["date_unregistration"].fillna(9999)

    # Base student master table
    master_df = pd.merge(student_info, courses, on=["code_module", "code_presentation"], how="left")
    master_df = pd.merge(master_df, student_reg, on=["code_module", "code_presentation", "id_student"], how="left")

    # Assessment processing
    student_assess_features, assess_item_analytics = process_assessments(raw_dir)
    master_df = pd.merge(master_df, student_assess_features, on=["code_module", "code_presentation", "id_student"], how="left")

    # VLE processing
    vle_features = process_vle(raw_dir)
    master_df = pd.merge(master_df, vle_features, on=["code_module", "code_presentation", "id_student"], how="left")

    # Handle Missing Values in analytical features
    master_df["total_assessments_submitted"] = master_df["total_assessments_submitted"].fillna(0)
    master_df["avg_assessment_score"] = master_df["avg_assessment_score"].fillna(0.0)
    master_df["median_assessment_score"] = master_df["median_assessment_score"].fillna(0.0)
    master_df["std_assessment_score"] = master_df["std_assessment_score"].fillna(0.0)
    master_df["min_assessment_score"] = master_df["min_assessment_score"].fillna(0.0)
    master_df["max_assessment_score"] = master_df["max_assessment_score"].fillna(0.0)
    master_df["weighted_score"] = master_df["weighted_score"].fillna(0.0)
    master_df["total_late_submissions"] = master_df["total_late_submissions"].fillna(0)
    master_df["late_submission_rate"] = master_df["late_submission_rate"].fillna(0.0)
    master_df["avg_submission_delay"] = master_df["avg_submission_delay"].fillna(0.0)
    master_df["assessment_completion_rate"] = master_df["assessment_completion_rate"].fillna(0.0)
    master_df["tma_avg_score"] = master_df["tma_avg_score"].fillna(0.0)
    master_df["cma_avg_score"] = master_df["cma_avg_score"].fillna(0.0)
    master_df["exam_score"] = master_df["exam_score"].fillna(0.0)

    master_df["total_vle_clicks"] = master_df["total_vle_clicks"].fillna(0)
    master_df["vle_active_days"] = master_df["vle_active_days"].fillna(0)
    master_df["clicks_per_active_day"] = master_df["clicks_per_active_day"].fillna(0.0)
    master_df["pre_course_clicks"] = master_df["pre_course_clicks"].fillna(0)
    master_df["early_course_clicks"] = master_df["early_course_clicks"].fillna(0)
    master_df["mid_course_clicks"] = master_df["mid_course_clicks"].fillna(0)
    master_df["late_course_clicks"] = master_df["late_course_clicks"].fillna(0)
    master_df["click_forum"] = master_df["click_forum"].fillna(0)
    master_df["click_oucontent"] = master_df["click_oucontent"].fillna(0)
    master_df["click_quiz"] = master_df["click_quiz"].fillna(0)
    master_df["click_resource"] = master_df["click_resource"].fillna(0)
    master_df["click_subpage"] = master_df["click_subpage"].fillna(0)
    master_df["click_url"] = master_df["click_url"].fillna(0)
    master_df["click_homepage"] = master_df["click_homepage"].fillna(0)

    # Impute missing demographics
    master_df["imd_band"] = master_df["imd_band"].fillna("Missing")

    # Define Target Variables (Slide 3 & Slide 5)
    # Binary: Pass (Pass / Distinction) = 1, Fail (Fail / Withdrawn) = 0
    master_df["final_result_binary"] = master_df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    # At-risk indicator: 1 if Fail or Withdrawn
    master_df["is_at_risk"] = master_df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    # Save to Lakehouse Parquet & CSV formats
    print(f"[ETL] Saving processed dataset ({len(master_df)} records) to {output_dir}...")
    master_parquet = os.path.join(output_dir, "student_features.parquet")
    master_csv = os.path.join(output_dir, "student_features.csv")
    master_df.to_parquet(master_parquet, index=False)
    master_df.to_csv(master_csv, index=False)

    assess_analytics_parquet = os.path.join(output_dir, "assessment_analytics.parquet")
    assess_item_analytics.to_parquet(assess_analytics_parquet, index=False)
    assess_item_analytics.to_csv(os.path.join(output_dir, "assessment_analytics.csv"), index=False)

    print(f"[ETL] Pipeline finished successfully in {time.time() - start_total:.1f}s!")
    print(f"  - Master Features: {master_parquet} ({os.path.getsize(master_parquet)/1024/1024:.2f} MB)")
    print(f"  - Assessment Analytics: {assess_analytics_parquet} ({os.path.getsize(assess_analytics_parquet)/1024:.2f} KB)")

    return master_df


if __name__ == "__main__":
    build_analytical_dataset()
