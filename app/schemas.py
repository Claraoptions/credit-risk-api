"""Request and response models. Allowed values mirror the German Credit dataset."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StatusAccount = Literal["< 0 DM", "0 to < 200 DM", ">= 200 DM", "no checking account"]
CreditHistory = Literal[
    "no credits taken/ all credits paid back duly",
    "all credits at this bank paid back duly",
    "existing credits paid back duly till now",
    "delay in paying off in the past",
    "critical account/ other credits existing (not at this bank)",
]
Purpose = Literal[
    "car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances",
    "repairs", "education", "retraining", "business", "others",
]
StatusSavings = Literal[
    "< 100 DM", "100 to < 500 DM", "500 to < 1000 DM", ">= 1000 DM", "unknown/ no savings account",
]
YearsEmployment = Literal["unemployed", "< 1 year", "1 to < 4 years", "4 to < 7 years", ">= 7 years"]
SecondaryObligor = Literal["none", "co-applicant", "guarantor"]
Collateral = Literal["real estate", "savings agreement/life insurance", "car", "none"]
OtherInstallmentPlans = Literal["bank", "stores", "none"]
Housing = Literal["rent", "own", "for free"]
Job = Literal[
    "unemployed/ unskilled - non-resident",
    "unskilled - resident",
    "skilled employee/ official",
    "management/ self-employed/highly qualified employee",
]
Telephone = Literal["none", "yes, registered under the customers name"]


class Applicant(BaseModel):
    """A loan application. Protected attributes (sex, marital status, nationality) are not accepted."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status_account": "0 to < 200 DM",
                "month_duration": 24,
                "credit_history": "existing credits paid back duly till now",
                "purpose": "car (used)",
                "credit_amount": 3000,
                "status_savings": "< 100 DM",
                "years_employment": "1 to < 4 years",
                "payment_to_income_ratio": 3,
                "secondary_obligor": "none",
                "residence_since": 2,
                "collateral": "car",
                "age": 35,
                "other_installment_plans": "none",
                "housing": "own",
                "n_credits": 1,
                "job": "skilled employee/ official",
                "n_guarantors": 1,
                "telephone": "none",
            }
        },
    )

    status_account: StatusAccount
    month_duration: int = Field(ge=1, le=120, description="Loan duration in months")
    credit_history: CreditHistory
    purpose: Purpose
    credit_amount: float = Field(gt=0, le=1_000_000, description="Requested amount (DM)")
    status_savings: StatusSavings
    years_employment: YearsEmployment
    payment_to_income_ratio: int = Field(ge=1, le=4, description="Instalment as % of income, banded 1 (low) to 4")
    secondary_obligor: SecondaryObligor
    residence_since: int = Field(ge=1, le=4)
    collateral: Collateral
    age: int = Field(ge=18, le=100)
    other_installment_plans: OtherInstallmentPlans
    housing: Housing
    n_credits: int = Field(ge=1, le=10, description="Existing credits at this bank")
    job: Job
    n_guarantors: int = Field(ge=1, le=5, description="Number of people liable to provide maintenance")
    telephone: Telephone


class Factor(BaseModel):
    feature: str
    contribution: float = Field(description="Effect on the log-odds of default; positive raises risk")


class Prediction(BaseModel):
    probability_of_default: float
    risk_band: Literal["Low", "Medium", "High"]
    recommendation: Literal["Approve", "Refer to credit officer", "Decline"]
    top_risk_factors: list[Factor]
    model_version: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_version: str
    algorithm: str
    trained_on_rows: int
    features: list[str]
    excluded_features: list[str]
    cv_roc_auc: float
    test_roc_auc: float
    test_recall_default: float
    risk_band_thresholds: dict[str, float]
