"""
End-to-End Pipeline Execution Script
Runs the complete workflow:
1. Verifies/Downloads OULAD raw data
2. Executes Big Data / Lakehouse ETL pipeline
3. Generates Statistical tests and Curriculum Analytics
4. Trains and evaluates ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
5. Validates SHAP explainability
6. Runs pytest suite
"""

import os
import sys
import time
import subprocess

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def step(title: str):
    print(f"\n{'='*70}\n[STEP] {title}\n{'='*70}")


def run_pipeline():
    start_all = time.time()

    # Step 1: Check / Download Data
    step("1. Verifying OULAD Raw Datasets")
    if not os.path.exists("data/raw/studentVle.csv"):
        print("Data not found. Running download...")
        subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
    else:
        print("Raw datasets verified in data/raw/.")

    # Step 2: Lakehouse Vectorized ETL
    step("2. Running Lakehouse ETL & Feature Engineering Pipeline")
    from src.etl.data_pipeline import build_analytical_dataset
    build_analytical_dataset()

    # Step 3: Statistical Analytics
    step("3. Computing Curriculum Analytics & Statistical Tests")
    from src.analytics.curriculum_analytics import get_course_performance_summary
    from src.analytics.statistical_tests import run_statistical_analysis
    perf = get_course_performance_summary()
    stats = run_statistical_analysis()
    print(f"Overall Pass Rate: {perf['overall_kpis']['overall_pass_rate']}%")
    print(f"OLS Regression R2: {stats['ols_regression']['r_squared']}")

    # Step 4: Machine Learning Model Training
    step("4. Training & Evaluating ML Models")
    from src.ml.train_models import train_and_evaluate_all_models
    train_and_evaluate_all_models()

    # Step 5: Explainable AI Verification
    step("5. Validating Explainable AI & SHAP Pipeline")
    from src.ml.explainability import ModelExplainer
    explainer = ModelExplainer()
    global_exp = explainer.get_global_explanations()
    print(f"Top 3 Global Driving Features: {[f['feature'] for f in global_exp['top_factors'][:3]]}")

    # Step 6: Test Suite
    step("6. Running Pytest Verification Suite")
    ret = subprocess.run([sys.executable, "-m", "pytest", "tests/"])
    if ret.returncode != 0:
        print("[Warning] Some tests failed. Please review logs.")
    else:
        print("[Success] All test suites passed!")

    print(f"\n{'='*70}\n[COMPLETE] Full Pipeline Finished in {time.time() - start_all:.1f}s!\n{'='*70}")


if __name__ == "__main__":
    run_pipeline()
