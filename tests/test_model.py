import os
import pickle
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.model import train_model, save_model_pickle, compute_model_metrics


@pytest.mark.parametrize(
    "X_train, y_train",
    [
        (np.array([[1, 2], [2, 3], [3, 4], [4, 5]]), np.array([0, 1, 0, 1])),
        (np.array([[0, 1], [1, 0], [0, 1], [1, 0]]), np.array([0, 1, 0, 1])),
        (np.array([[2, 3], [3, 4], [4, 5], [5, 6]]), np.array([1, 0, 1, 0])),
    ]
)
def test_train_model(X_train, y_train):
    # Train the model
    model = train_model(X_train, y_train)

    # Model is LR and fitted
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")


@pytest.mark.parametrize(
    "X_train, y_train",
    [
        (np.array([[1, 2], [2, 3], [3, 4], [4, 5]]), np.array([0, 1, 0, 1])),
        (np.array([[0, 1], [1, 0], [0, 1], [1, 0]]), np.array([0, 1, 0, 1])),
        (np.array([[2, 3], [3, 4], [4, 5], [5, 6]]), np.array([1, 0, 1, 0])),
    ]
)
def test_save_model_pickle(X_train, y_train, tmpdir):
    # Create a model
    model = train_model(X_train, y_train)

    # Save the model
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model.pkl')
        save_model_pickle(model, model_path)
        assert os.path.exists(model_path)

        # Check that if we load the model, it is the same
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
            assert isinstance(loaded_model, LogisticRegression)
            assert np.array_equal(model.coef_, loaded_model.coef_)


@patch('ml.model.precision_score')
@patch('ml.model.recall_score')
@patch('ml.model.fbeta_score')
def test_compute_model_metrics(mock_fbeta_score, mock_recall_score, mock_precision_score):
    # Create data and call the tested function
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check if the mocked functions were called and with the expected parameters
    mock_precision_score.assert_called_once_with(y_true, y_pred, zero_division=1)
    mock_recall_score.assert_called_once_with(y_true, y_pred, zero_division=1)
    mock_fbeta_score.assert_called_once_with(y_true, y_pred, beta=1, zero_division=1)

    # Assertions
    assert precision == mock_precision_score.return_value
    assert recall == mock_recall_score.return_value
    assert fbeta == mock_fbeta_score.return_value
