import joblib
import pytest

from app import model as m


@pytest.mark.parametrize("p,band", [(0.0, "Low"), (0.349, "Low"), (0.35, "Medium"), (0.599, "Medium"), (0.6, "High")])
def test_risk_band_boundaries(p, band):
    assert m.risk_band(p) == band


def test_training_metrics_use_held_out_data():
    trained = m.train()
    assert trained.rows == 1000
    assert set(trained.metrics) == {"cv_roc_auc", "test_roc_auc", "test_recall_default"}
    assert trained.metrics["test_roc_auc"] > 0.70


def test_protected_attributes_are_not_features():
    assert not set(m.EXCLUDED) & set(m.FEATURES)


def test_load_or_train_creates_and_reuses_model(tmp_path):
    path = tmp_path / "model.joblib"
    first = m.load_or_train(path)
    assert path.exists()
    assert m.load_or_train(path).metrics == first.metrics


def test_load_or_train_retrains_stale_version(tmp_path):
    path = tmp_path / "model.joblib"
    stale = m.train()
    stale.version = "0.0.1"
    joblib.dump(stale, path)
    assert m.load_or_train(path).version == m.MODEL_VERSION
