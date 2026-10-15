"""
Generates the comprehensive Jupyter Notebook for Curriculum Effectiveness Analysis.
"""

import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Big Data Architecture: Curriculum Effectiveness Analysis & Student Outcome Prediction\n",
    "### Based on the Open University Learning Analytics Dataset (OULAD)\n",
    "\n",
    "This research and experimentation notebook implements the end-to-end Big Data learning analytics lifecycle:\n",
    "1. **Data Ingestion & Distributed Lakehouse Architecture** (7 relational tables, 10.6M VLE interaction records)\n",
    "2. **Level 1: Curriculum Effectiveness Analytics** (Course benchmarks, TMA vs CMA vs Exam, VLE engagement dynamics)\n",
    "3. **Statistical Analysis & Hypothesis Testing** (Pearson/Spearman correlation matrices, OLS regression, ANOVA tests)\n",
    "4. **Level 2: Student Academic Outcome Prediction** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)\n",
    "5. **Explainable AI with SHAP** (Global feature importance & local individual waterfall explanations)\n",
    "6. **Actionable Pedagogical Interventions** (Automated advisor alerts & curriculum review recommendations)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import shap\n",
    "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report\n",
    "\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "plt.rcParams['figure.figsize'] = (10, 5)\n",
    "print('All libraries loaded successfully.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Processed Lakehouse Feature Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_parquet('../data/processed/student_features.parquet')\n",
    "print(f'Total Student Records: {len(df):,}')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Course-Level Benchmark: Pass vs Attrition vs Failure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.analytics.curriculum_analytics import get_course_performance_summary\n",
    "summary = get_course_performance_summary(df)\n",
    "modules_df = pd.DataFrame(summary['modules'])\n",
    "print(modules_df[['code_module', 'enrolled', 'pass_rate', 'withdrawal_rate', 'fail_rate', 'avg_score']])\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "modules_df.set_index('code_module')[['pass_rate', 'withdrawal_rate', 'fail_rate']].plot(kind='bar', ax=ax)\n",
    "plt.title('Curriculum Module Benchmark: Pass, Withdrawal, and Fail Rates')\n",
    "plt.ylabel('Percentage (%)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Assessment Effectiveness Analysis: TMA vs CMA vs Exam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.analytics.assessment_analytics import get_assessment_effectiveness_summary\n",
    "assess_summary = get_assessment_effectiveness_summary()\n",
    "type_df = pd.DataFrame(assess_summary['assessment_type_summary'])\n",
    "print(type_df)\n",
    "\n",
    "# Challenging Bottleneck Assessments\n",
    "bottlenecks_df = pd.DataFrame(assess_summary['challenging_assessments'])\n",
    "print('\\nTop Curriculum Bottlenecks (High Failure/Late Submissions):')\n",
    "print(bottlenecks_df[['id_assessment', 'code_module', 'assessment_type', 'weight', 'avg_score', 'pass_rate', 'difficulty_index']].head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Student Engagement Analysis (VLE Clickstreams)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.analytics.engagement_analytics import get_engagement_summary\n",
    "eng_summary = get_engagement_summary()\n",
    "tiers_df = pd.DataFrame(eng_summary['active_days_tiers'])\n",
    "print(tiers_df)\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "sns.barplot(data=tiers_df, x='active_days_tier', y='pass_rate', palette='viridis')\n",
    "plt.title('Impact of Active Study Days on Passing Probability')\n",
    "plt.ylabel('Pass Rate (%)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Statistical Analysis & Multiple Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.analytics.statistical_tests import run_statistical_analysis\n",
    "stats_results = run_statistical_analysis()\n",
    "print('OLS Model R-Squared:', stats_results['ols_regression']['r_squared'])\n",
    "print('\\nSignificant Pedagogical Findings:')\n",
    "for f in stats_results['curriculum_findings']:\n",
    "    print(f'- {f[\"finding\"]}: {f[\"evidence\"]}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Machine Learning Model Evaluation & Comparisons"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "with open('../data/models/model_metrics.json') as f:\n",
    "    metrics = json.load(f)\n",
    "\n",
    "comparison_df = pd.DataFrame([\n",
    "    {'Model': name, **m}\n",
    "    for name, m in metrics['models'].items()\n",
    "]).drop(columns=['confusion_matrix'])\n",
    "\n",
    "print(comparison_df.to_string(index=False))\n",
    "\n",
    "# Plot Model ROC-AUC Comparison\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.barplot(data=comparison_df, x='Model', y='roc_auc', palette='Blues_r')\n",
    "plt.title('ML Models ROC-AUC Comparison')\n",
    "plt.ylim(0.9, 1.0)\n",
    "plt.ylabel('ROC-AUC Score')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Model Explainability with SHAP (Global & Individual)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.ml.explainability import ModelExplainer\n",
    "explainer = ModelExplainer(model_dir='../data/models')\n",
    "global_shap = explainer.get_global_explanations()\n",
    "\n",
    "shap_df = pd.DataFrame(global_shap['top_factors'])\n",
    "print('Global Feature Importance Ranking:')\n",
    "print(shap_df.head(10))\n",
    "\n",
    "# Individual Simulation with Tailored Interventions\n",
    "student_profile = {\n",
    "    'code_module': 'BBB',\n",
    "    'gender': 'F',\n",
    "    'highest_education': 'A Level or Equivalent',\n",
    "    'age_band': '0-35',\n",
    "    'disability': 'N',\n",
    "    'studied_credits': 60,\n",
    "    'num_of_prev_attempts': 0,\n",
    "    'days_before_course_registered': 30,\n",
    "    'total_assessments_submitted': 2,\n",
    "    'avg_assessment_score': 48.0,\n",
    "    'weighted_score': 35.0,\n",
    "    'late_submission_rate': 0.5,\n",
    "    'avg_submission_delay': 4.0,\n",
    "    'assessment_completion_rate': 0.4,\n",
    "    'tma_submitted': 2,\n",
    "    'tma_avg_score': 48.0,\n",
    "    'cma_submitted': 0,\n",
    "    'cma_avg_score': 0.0,\n",
    "    'total_vle_clicks': 320,\n",
    "    'vle_active_days': 16,\n",
    "    'clicks_per_active_day': 20.0,\n",
    "    'pre_course_clicks': 0,\n",
    "    'early_course_clicks': 150,\n",
    "    'click_forum': 40,\n",
    "    'click_oucontent': 180,\n",
    "    'click_quiz': 15,\n",
    "    'click_resource': 25,\n",
    "    'click_url': 10\n",
    "}\n",
    "\n",
    "explanation = explainer.explain_individual(student_profile)\n",
    "print('\\n--- PREDICTION RESULT ---')\n",
    "print('Outcome:', explanation['prediction']['outcome'])\n",
    "print('Risk Tier:', explanation['prediction']['risk_tier'])\n",
    "print('Fail Probability:', explanation['prediction']['probability_fail'], '%')\n",
    "print('\\n--- RECOMMENDED ADVISOR INTERVENTIONS ---')\n",
    "for action in explanation['recommended_interventions']:\n",
    "    print(f'[{action[\"priority\"]} Priority] {action[\"type\"]}: {action[\"action\"]}')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('notebooks/curriculum_analysis_experimentation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Generated notebooks/curriculum_analysis_experimentation.ipynb successfully.")
