"""
Curriculum Effectiveness Analysis - PySpark Distributed Big Data Pipeline
Implements the distributed architecture depicted in Slide 3:
- Ingestion of OULAD relational datasets from HDFS / Lakehouse
- Distributed cleaning, joins, aggregations via Spark DataFrame & SQL
- PySpark Feature Engineering & Vector Assembler for Spark MLlib
"""

import os
from typing import Optional

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
    from pyspark.ml import Pipeline
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def create_spark_session(app_name: str = "CurriculumEffectivenessAnalysis") -> Optional["SparkSession"]:
    """Initializes a local or cluster SparkSession with optimized memory settings."""
    if not PYSPARK_AVAILABLE:
        print("[Spark] PySpark is not installed in the current environment. To run distributed jobs, install pyspark.")
        return None

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_spark_etl(raw_data_dir: str = "data/raw", output_dir: str = "data/processed/spark"):
    """
    Executes distributed PySpark ETL:
    1. Reads raw CSV tables into distributed Spark DataFrames
    2. Performs distributed aggregations on 10.6M VLE interaction logs
    3. Calculates student assessment metrics (TMA, CMA, exam performance, late submissions)
    4. Joins student registration and demographics
    5. Saves partitioned Parquet files for distributed analytics
    """
    if not PYSPARK_AVAILABLE:
        print("[Spark] Skipping PySpark execution: PySpark not available.")
        return

    spark = create_spark_session()
    if not spark:
        return

    print("[Spark] Starting distributed ETL job...")
    
    # 1. Read Raw Datasets
    student_info = spark.read.csv(os.path.join(raw_data_dir, "studentInfo.csv"), header=True, inferSchema=True)
    assessments = spark.read.csv(os.path.join(raw_data_dir, "assessments.csv"), header=True, inferSchema=True)
    student_assess = spark.read.csv(os.path.join(raw_data_dir, "studentAssessment.csv"), header=True, inferSchema=True)
    student_vle = spark.read.csv(os.path.join(raw_data_dir, "studentVle.csv"), header=True, inferSchema=True)
    vle = spark.read.csv(os.path.join(raw_data_dir, "vle.csv"), header=True, inferSchema=True)
    courses = spark.read.csv(os.path.join(raw_data_dir, "courses.csv"), header=True, inferSchema=True)
    student_reg = spark.read.csv(os.path.join(raw_data_dir, "studentRegistration.csv"), header=True, inferSchema=True)

    # 2. Distributed VLE Aggregation
    print("[Spark] Aggregating 10.6M VLE click events across cluster...")
    vle_joined = student_vle.join(vle.select("id_site", "activity_type"), on="id_site", how="left")

    vle_agg = vle_joined.groupBy("code_module", "code_presentation", "id_student").agg(
        F.sum("sum_click").alias("total_vle_clicks"),
        F.countDistinct("date").alias("vle_active_days"),
        F.sum(F.when(F.col("date") < 0, F.col("sum_click")).otherwise(0)).alias("pre_course_clicks"),
        F.sum(F.when((F.col("date") >= 0) & (F.col("date") <= 30), F.col("sum_click")).otherwise(0)).alias("early_course_clicks"),
        F.sum(F.when(F.col("activity_type") == "forumng", F.col("sum_click")).otherwise(0)).alias("click_forum"),
        F.sum(F.when(F.col("activity_type") == "oucontent", F.col("sum_click")).otherwise(0)).alias("click_oucontent"),
        F.sum(F.when(F.col("activity_type").isin(["quiz", "externalquiz"]), F.col("sum_click")).otherwise(0)).alias("click_quiz"),
        F.sum(F.when(F.col("activity_type") == "resource", F.col("sum_click")).otherwise(0)).alias("click_resource")
    )

    # 3. Assessment Features
    print("[Spark] Processing student assessment performance and submission behavior...")
    assess_joined = student_assess.join(assessments, on="id_assessment", how="inner")
    assess_with_delay = assess_joined.withColumn(
        "submission_delay",
        F.col("date_submitted") - F.coalesce(F.col("date"), F.col("date_submitted"))
    ).withColumn(
        "is_late",
        F.when(F.col("submission_delay") > 0, 1).otherwise(0)
    ).withColumn(
        "weighted_score_contrib",
        F.col("score") * (F.col("weight") / 100.0)
    )

    assess_agg = assess_with_delay.groupBy("code_module", "code_presentation", "id_student").agg(
        F.count("id_assessment").alias("total_assessments_submitted"),
        F.avg("score").alias("avg_assessment_score"),
        F.sum("weighted_score_contrib").alias("weighted_score"),
        F.sum("is_late").alias("total_late_submissions"),
        F.avg("is_late").alias("late_submission_rate"),
        F.avg("submission_delay").alias("avg_submission_delay"),
        F.avg(F.when(F.col("assessment_type") == "TMA", F.col("score"))).alias("tma_avg_score"),
        F.avg(F.when(F.col("assessment_type") == "CMA", F.col("score"))).alias("cma_avg_score"),
        F.avg(F.when(F.col("assessment_type") == "Exam", F.col("score"))).alias("exam_score")
    )

    # 4. Integrate Master Analytics Table
    print("[Spark] Integrating master analytical dataset...")
    master_spark = (
        student_info.join(courses, on=["code_module", "code_presentation"], how="left")
        .join(student_reg, on=["code_module", "code_presentation", "id_student"], how="left")
        .join(assess_agg, on=["code_module", "code_presentation", "id_student"], how="left")
        .join(vle_agg, on=["code_module", "code_presentation", "id_student"], how="left")
    )

    # Target variable definitions
    master_spark = master_spark.withColumn(
        "final_result_binary",
        F.when(F.col("final_result").isin(["Pass", "Distinction"]), 1).otherwise(0)
    ).withColumn(
        "is_at_risk",
        F.when(F.col("final_result").isin(["Fail", "Withdrawn"]), 1).otherwise(0)
    )

    # Save to Parquet
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "student_features_parquet")
    print(f"[Spark] Writing partitioned Parquet dataset to {output_path}...")
    master_spark.write.mode("overwrite").parquet(output_path)
    print("[Spark] PySpark ETL completed successfully!")
    spark.stop()


if __name__ == "__main__":
    run_spark_etl()
