"""
Machine Learning Training Module
Trains and compares the 4 models specified in Slide 3 & Slide 5:
  1. Logistic Regression (Baseline)
  2. Decision Tree (Interpretable)
  3. Random Forest (Main Ensemble Model)
  4. Gradient Boosting Classifier (High-Performance Ensemble)
Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

FEATURE_COLS_NUMERIC = [
    "studied_credits",
    "num_of_prev_attempts",
    "days_before_course_registered",
    "total_assessments_submitted",
    "avg_assessment_score",
    "weighted_score",
    "late_submission_rate",
    "avg_submission_delay",
    "assessment_completion_rate",
    "tma_submitted",
    "tma_avg_score",
    "cma_submitted",
    "cma_avg_score",
    "total_vle_clicks",
    "vle_active_days",
    "clicks_per_active_day",
    "pre_course_clicks",
    "early_course_clicks",
    "click_forum",
    "click_oucontent",
    "click_quiz",
    "click_resource",
    "click_url"
]

FEATURE_COLS_CATEGORICAL = [
    "code_module",
    "gender",
    "highest_education",
    "age_band",
    "disability"
]

ALL_FEATURE_COLS = FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL
TARGET_COL = "final_result_binary"


def load_dataset(path: str = "data/processed/student_features.parquet") -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.read_csv("data/processed/student_features.csv")


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLS_NUMERIC),
            ("cat", categorical_transformer, FEATURE_COLS_CATEGORICAL)
        ]
    )
    return preprocessor


def train_and_evaluate_all_models(data_path: str = "data/processed/student_features.parquet", output_dir: str = "data/models") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(data_path)
    print(f"[ML] Training dataset loaded: {len(df)} records.")

    # Prepare features and labels
    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET_COL].values

    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract feature names after One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(FEATURE_COLS_CATEGORICAL))
    transformed_feature_names = FEATURE_COLS_NUMERIC + encoded_cat_names

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    metrics_results = {}
    fitted_models = {}
    best_model_name = None
    best_f1 = -1.0

    print("[ML] Training models and computing metrics...")
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train_proc, y_train)
        fit_time = round(time.time() - t0, 2)

        # Predictions
        y_pred = clf.predict(X_test_proc)
        y_prob = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics
        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        roc_auc = round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else 0.0

        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "training_time_seconds": fit_time
        }
        fitted_models[name] = clf

        print(f"  - {name}: Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f} (Fit time: {fit_time}s)")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"[ML] Best Model: {best_model_name} (F1 Score: {best_f1:.4f})")

    # Save artifacts
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(fitted_models[best_model_name], os.path.join(output_dir, "best_model.joblib"))
    joblib.dump(fitted_models, os.path.join(output_dir, "all_models.joblib"))

    # Save transformed feature names
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(transformed_feature_names, f, indent=2)

    # Save evaluation summary
    evaluation_summary = {
        "best_model_name": best_model_name,
        "models": metrics_results,
        "test_sample_size": int(len(y_test)),
        "train_sample_size": int(len(y_train)),
        "feature_count": len(transformed_feature_names),
        "target_distribution": {
            "pass_ratio": round(float(y.mean()), 4),
            "fail_ratio": round(float(1.0 - y.mean()), 4)
        }
    }

    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    # Save representative background sample for SHAP explainability
    sample_background = X_train.sample(n=200, random_state=42)
    sample_background.to_parquet(os.path.join(output_dir, "shap_background.parquet"), index=False)

    print(f"[ML] Models and metrics successfully saved to {output_dir}/")
    return evaluation_summary


if __name__ == "__main__":
    train_and_evaluate_all_models()
