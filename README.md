# Credit Risk Scoring API

[![CI](https://github.com/Claraoptions/credit-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Claraoptions/credit-risk-api/actions/workflows/ci.yml)

**Ayebazibwe Clare Mujuni | Software Development Portfolio**

A REST API that scores loan applications for **probability of default**, assigns a **risk band** and **recommendation**, and explains the **main risk drivers** behind each score. Built with FastAPI and scikit-learn, packaged with Docker, and tested on every push.

> Educational project trained on the public German Credit dataset (1,000 applications). It is not a lending decision tool.

---

## What it does

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check for load balancers and containers |
| `GET /model/info` | Model version, features used, features excluded, and held-out performance |
| `POST /predict` | Score one application |
| `POST /predict/batch` | Score up to 500 applications in one call |
| `GET /docs` | Interactive Swagger documentation |

Example response:

```json
{
  "probability_of_default": 0.4366,
  "risk_band": "Medium",
  "recommendation": "Refer to credit officer",
  "top_risk_factors": [
    {"feature": "status_savings", "contribution": 0.476},
    {"feature": "status_account", "contribution": 0.339},
    {"feature": "secondary_obligor", "contribution": 0.187}
  ],
  "model_version": "1.0.0"
}
```

| Probability of default | Risk band | Recommendation |
|---|---|---|
| below 0.35 | Low | Approve |
| 0.35 – 0.60 | Medium | Refer to credit officer |
| 0.60 and above | High | Decline |

## Design decisions

These come from my years making and auditing credit decisions in banking:

- **Protected attributes are excluded.** The dataset includes sex/marital status and foreign-worker status. The model does not use them, and the API rejects requests that send them, so they cannot influence a decision.
- **Explainable by design.** A class-balanced logistic regression lets every score be broken down into per-feature contributions, so a credit officer can see *why* an application was referred or declined.
- **Defaulters matter most.** Approving a bad loan costs far more than reviewing a good one, so the model is class-balanced and reports recall on defaulters, not just accuracy.
- **Honest metrics.** Cross-validation and test scores come from data the model did not see in training; the final model is then refit on all rows.
- **Strict input validation.** Every categorical field accepts only known values, numbers have sensible ranges, and unknown fields are rejected (HTTP 422).

## Model performance

| Metric | Value |
|---|---|
| 5-fold cross-validated ROC-AUC (training split) | 0.765 |
| Held-out test ROC-AUC (200 applications) | 0.81 |
| Held-out recall on defaulters | 0.82 |

## Run it

### Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Python

```bash
pip install -r requirements.txt
python -m app.train          # optional: the API trains on first start if no model exists
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs, or:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "status_account": "0 to < 200 DM", "month_duration": 24,
  "credit_history": "existing credits paid back duly till now", "purpose": "car (used)",
  "credit_amount": 3000, "status_savings": "< 100 DM", "years_employment": "1 to < 4 years",
  "payment_to_income_ratio": 3, "secondary_obligor": "none", "residence_since": 2,
  "collateral": "car", "age": 35, "other_installment_plans": "none", "housing": "own",
  "n_credits": 1, "job": "skilled employee/ official", "n_guarantors": 1, "telephone": "none"
}'
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest --cov=app
```

20 tests cover the endpoints, input validation (including rejection of protected attributes), risk-band boundaries, model training and model reloading. CI runs them on Python 3.10–3.12, then builds the Docker image and smoke-tests the running container.

## Project structure

```
credit-risk-api/
├── app/
│   ├── main.py      # FastAPI app and endpoints
│   ├── schemas.py   # request/response models and allowed values
│   ├── model.py     # training, evaluation, scoring and explanations
│   └── train.py     # python -m app.train
├── data/german_credit_data.csv
├── tests/
├── Dockerfile
└── .github/workflows/ci.yml
```

Related analysis: [German credit risk classification notebook](https://github.com/Claraoptions/German_credit_prediction).

## Author

**Ayebazibwe Clare Mujuni** — 20 years in banking across branch management, credit risk and internal audit; now building data and software tools for finance.
[GitHub](https://github.com/Claraoptions) · [LinkedIn](https://www.linkedin.com/in/ayebazibwe-clare-mujuni-91212a240) · mujuniclare@gmail.com

## License

MIT
