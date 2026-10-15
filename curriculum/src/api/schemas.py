"""
Pydantic Request & Response Schemas for Curriculum Effectiveness Analysis API
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StudentPredictionRequest(BaseModel):
    code_module: str = Field(default="BBB", description="Course module code (AAA-GGG)")
    gender: str = Field(default="F", description="Gender (M/F)")
    highest_education: str = Field(default="A Level or Equivalent", description="Highest education level")
    age_band: str = Field(default="0-35", description="Age band (0-35, 35-55, 55<=)")
    disability: str = Field(default="N", description="Disability status (Y/N)")
    studied_credits: int = Field(default=60, ge=30, le=360, description="Total credits studying")
    num_of_prev_attempts: int = Field(default=0, ge=0, le=10, description="Previous attempts on this course")
    days_before_course_registered: int = Field(default=30, description="Days registered prior to start")
    total_assessments_submitted: int = Field(default=4, ge=0, le=20, description="Number of assessments submitted")
    avg_assessment_score: float = Field(default=68.5, ge=0.0, le=100.0, description="Average assessment score")
    weighted_score: float = Field(default=65.0, ge=0.0, le=100.0, description="Cumulative weighted score")
    late_submission_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of late submissions")
    avg_submission_delay: float = Field(default=-1.5, description="Average days before/after deadline")
    assessment_completion_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="Completion percentage")
    tma_submitted: int = Field(default=3, ge=0, description="Tutor-marked assessments submitted")
    tma_avg_score: float = Field(default=70.0, ge=0.0, le=100.0, description="Average TMA score")
    cma_submitted: int = Field(default=1, ge=0, description="Computer-marked assessments submitted")
    cma_avg_score: float = Field(default=65.0, ge=0.0, le=100.0, description="Average CMA score")
    total_vle_clicks: int = Field(default=1200, ge=0, description="Total clicks across VLE")
    vle_active_days: int = Field(default=48, ge=0, le=300, description="Distinct active study days")
    clicks_per_active_day: float = Field(default=25.0, ge=0.0, description="Average clicks per active day")
    pre_course_clicks: int = Field(default=100, ge=0, description="Clicks before module start")
    early_course_clicks: int = Field(default=350, ge=0, description="Clicks during first 30 days")
    click_forum: int = Field(default=300, ge=0, description="Discussion forum clicks")
    click_oucontent: int = Field(default=500, ge=0, description="Course content clicks")
    click_quiz: int = Field(default=120, ge=0, description="Interactive quiz clicks")
    click_resource: int = Field(default=80, ge=0, description="PDF/resource clicks")
    click_url: int = Field(default=40, ge=0, description="External URL clicks")


class PredictionResponse(BaseModel):
    prediction: Dict[str, Any]
    top_contributing_factors: List[Dict[str, Any]]
    factors_pushing_pass: List[Dict[str, Any]]
    factors_pushing_fail: List[Dict[str, Any]]
    recommended_interventions: List[Dict[str, str]]
