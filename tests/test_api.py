import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Applicant

EXAMPLE = Applicant.model_config["json_schema_extra"]["example"]

LOW_RISK = {
    **EXAMPLE,
    "status_account": "no checking account",
    "credit_history": "critical account/ other credits existing (not at this bank)",
    "month_duration": 12,
    "credit_amount": 1500,
    "status_savings": ">= 1000 DM",
    "years_employment": ">= 7 years",
    "collateral": "real estate",
    "age": 45,
}
HIGH_RISK = {
    **EXAMPLE,
    "status_account": "< 0 DM",
    "credit_history": "no credits taken/ all credits paid back duly",
    "month_duration": 60,
    "credit_amount": 15000,
    "status_savings": "< 100 DM",
    "years_employment": "< 1 year",
    "payment_to_income_ratio": 4,
    "collateral": "none",
    "housing": "rent",
    "age": 21,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_reports_holdout_metrics_and_exclusions(client):
    info = client.get("/model/info").json()
    assert info["trained_on_rows"] == 1000
    assert "status_and_sex" in info["excluded_features"]
    assert "status_and_sex" not in info["features"]
    assert info["test_roc_auc"] > 0.70
    assert info["cv_roc_auc"] > 0.70


def test_predict_returns_valid_prediction(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert len(body["top_risk_factors"]) <= 3
    assert all(f["contribution"] > 0 for f in body["top_risk_factors"])


def test_riskier_profile_scores_higher(client):
    low = client.post("/predict", json=LOW_RISK).json()
    high = client.post("/predict", json=HIGH_RISK).json()
    assert high["probability_of_default"] > low["probability_of_default"]
    assert low["recommendation"] == "Approve"
    assert high["recommendation"] == "Decline"


def test_batch(client):
    r = client.post("/predict/batch", json={"applicants": [LOW_RISK, HIGH_RISK]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "holiday"},  # not an allowed category
        {"age": 15},  # out of range
        {"credit_amount": -10},
        {"status_and_sex": "male : single"},  # protected attribute rejected
    ],
)
def test_invalid_input_is_rejected(client, change):
    assert client.post("/predict", json={**EXAMPLE, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    body = dict(EXAMPLE)
    body.pop("age")
    assert client.post("/predict", json=body).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422
