"""
Tests for the ML-based trading strategy.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from tick_analysis.core.strategies.ml_strategy import MLStrategy


# Sample test data
@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))) * 100

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * (1 + np.random.uniform(0, 0.01, len(dates))),
            "low": prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            "close": prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    return df


@pytest.fixture
def trained_strategy(sample_ohlcv_data: pd.DataFrame) -> MLStrategy:
    """Create a trained ML strategy for testing."""
    strategy = MLStrategy(model_type="random_forest")
    features = strategy.create_features(sample_ohlcv_data)
    mock_model = MagicMock()
    mock_model.feature_names_in_ = features.columns.tolist()
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array(
        [1, -1, 0, 1, -1] * (len(features) // 5 + 1)
    )[: len(features)]
    mock_pipeline.predict_proba.return_value = np.column_stack(
        [
            np.random.uniform(0, 0.3, len(features)),  # proba for -1
            np.random.uniform(0, 0.3, len(features)),  # proba for 0
            np.random.uniform(0.4, 1.0, len(features)),  # proba for 1
        ]
    )
    mock_pipeline.named_steps = {"model": mock_model}
    strategy.pipeline = mock_pipeline
    return strategy


def test_create_features(sample_ohlcv_data):
    """Test feature creation."""
    strategy = MLStrategy()
    features = strategy.create_features(sample_ohlcv_data)

    # Check that features are created
    expected_columns = [
        "returns",
        "log_returns",
        "sma_5",
        "sma_20",
        "sma_50",
        "volatility_20",
        "momentum_5",
        "momentum_10",
        "volume_change",
        "volume_ma_5",
        "volume_ratio",
    ]

    for col in expected_columns:
        assert col in features.columns

    # Check no NaN values in features
    assert not features.isna().any().any()


def test_create_target(sample_ohlcv_data):
    """Test target creation."""
    strategy = MLStrategy()
    target = strategy.create_target(
        sample_ohlcv_data, forward_periods=5, threshold=0.01
    )

    # Check target values are in [-1, 0, 1]
    assert set(target.unique()).issubset({-1, 0, 1})

    # Check target has the same length as input data
    # (The method doesn't drop any rows, it just shifts the target)
    assert len(target) == len(sample_ohlcv_data)


def test_prepare_data(sample_ohlcv_data):
    """Test data preparation."""
    strategy = MLStrategy()
    features = strategy.create_features(sample_ohlcv_data)
    target = strategy.create_target(sample_ohlcv_data)

    X, y = strategy.prepare_data(features, target)

    # Check X and y have the same number of samples
    assert len(X) == len(y)

    # Check feature columns don't contain OHLCV
    assert not any(
        col in X.columns for col in ["open", "high", "low", "close", "volume"]
    )


def test_train(sample_ohlcv_data):
    """Test model training."""
    strategy = MLStrategy()
    with patch("sklearn.ensemble.RandomForestClassifier") as mock_rf:
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_rf.return_value = mock_model
        # Create target for the sample data
        y = strategy.create_target(sample_ohlcv_data)
        metrics = strategy.train(sample_ohlcv_data, y)
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert "classification_report" in metrics
        assert "feature_importances" in metrics
        assert strategy.pipeline is not None


def test_predict(trained_strategy, sample_ohlcv_data):
    """Test prediction."""
    features = trained_strategy.create_features(sample_ohlcv_data)
    predictions = trained_strategy.predict(features)

    # Check predictions are in [-1, 0, 1]
    assert set(predictions).issubset({-1, 0, 1})


def test_predict_proba(trained_strategy, sample_ohlcv_data):
    """Test probability prediction."""
    features = trained_strategy.create_features(sample_ohlcv_data)
    n_samples = features.shape[0]
    proba = np.random.uniform(0.1, 0.9, size=(n_samples, 3))
    proba = proba / proba.sum(axis=1, keepdims=True)
    predictions = np.random.choice([-1, 0, 1], size=n_samples)
    trained_strategy.pipeline.predict_proba.return_value = proba
    trained_strategy.pipeline.predict.return_value = predictions

    # Test predict_proba
    result = trained_strategy.predict_proba(features)

    # Check output shape
    assert result.shape == (n_samples, 3)  # 3 classes

    # Check probabilities sum to 1 (within numerical tolerance)
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-8)

    # Verify the pipeline's predict_proba was called with the correct features
    called_features = trained_strategy.pipeline.predict_proba.call_args[0][0]
    assert called_features.shape == features.shape
    assert all(called_features.columns == features.columns)


def test_generate_signals(trained_strategy, sample_ohlcv_data):
    """Test signal generation."""
    # Create a copy of the sample data to avoid modifying the fixture
    data = sample_ohlcv_data.copy()

    # Create features using the strategy
    strategy = MLStrategy()
    features = strategy.create_features(data)
    n_samples = len(features)

    # Create mock predictions with proper index and values
    mock_predictions = np.random.choice([-1, 0, 1], size=n_samples)

    # Create probabilities that sum to 1 for each sample
    proba = np.random.uniform(0.1, 0.9, size=(n_samples, 3))
    proba = proba / proba.sum(axis=1, keepdims=True)

    # Set the mock pipeline's predict and predict_proba
    trained_strategy.pipeline.predict.return_value = mock_predictions
    trained_strategy.pipeline.predict_proba.return_value = proba
    trained_strategy.pipeline.predict.reset_mock()
    trained_strategy.pipeline.predict_proba.reset_mock()

    # Test generate_signals with a threshold
    threshold = 0.6

    # Call generate_signals with the original OHLCV data
    signals = trained_strategy.generate_signals(data, threshold=threshold)

    # Check signals are in [-1, 0, 1]
    assert set(signals.unique()).issubset({-1, 0, 1})

    # Check that the pipeline's predict_proba was called at least once (implementation calls it twice for MagicMock)
    assert trained_strategy.pipeline.predict_proba.call_count >= 1
    print(f"predict_proba call count: {trained_strategy.pipeline.predict_proba.call_count}")
    called_features = trained_strategy.pipeline.predict_proba.call_args[0][0]
    assert called_features.shape == features.shape
    assert all(called_features.columns == features.columns)


def test_save_load_model(trained_strategy, tmp_path, monkeypatch):
    """Test model saving and loading."""
    # Create a mock for the model data that will be saved/loaded
    mock_model_data = {
        "pipeline": trained_strategy.pipeline,
        "feature_importances": trained_strategy.feature_importances_,
        "model_type": "random_forest",
        "model_params": {},
    }

    # Mock joblib.dump to avoid actual file I/O
    mock_dump = MagicMock()
    monkeypatch.setattr("joblib.dump", mock_dump)

    # Mock joblib.load to return our mock model data
    mock_load = MagicMock(return_value=mock_model_data)
    monkeypatch.setattr("joblib.load", mock_load)

    # Mock os.makedirs to avoid actual directory creation
    mock_makedirs = MagicMock()
    monkeypatch.setattr("os.makedirs", mock_makedirs)

    # Mock os.path.exists to always return True for our test file
    mock_exists = MagicMock(return_value=True)
    monkeypatch.setattr("os.path.exists", mock_exists)

    # Test save
    model_path = tmp_path / "test_model.joblib"
    trained_strategy.save_model(model_path)

    # Check joblib.dump was called with the expected arguments
    mock_dump.assert_called_once()
    assert "pipeline" in mock_dump.call_args[0][0]
    assert "model_type" in mock_dump.call_args[0][0]

    # Test load
    loaded_strategy = MLStrategy.load_model(model_path)

    # Check loaded model has the same type
    assert isinstance(loaded_strategy, MLStrategy)
    assert loaded_strategy.model_type == "random_forest"

    # Verify joblib.load was called with the correct path
    # Convert both paths to strings for comparison to handle WindowsPath vs string
    called_path = str(mock_load.call_args[0][0])
    expected_path = str(model_path)
    assert (
        called_path == expected_path
    ), f"Expected path {expected_path}, got {called_path}"
