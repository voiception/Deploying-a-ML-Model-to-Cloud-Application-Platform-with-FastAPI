import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

test_client = TestClient(app)


def test_welcome():
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML model inference API"}


valid_test_cases = [
    ({"age": 36, "workclass": "Private", "fnlgt": 484024, "education": "HS-grad", "education-num": 9,
      "marital-status": "Divorced", "occupation": "Machine-op-inspct", "relationship": "Unmarried", "race": "White",
      "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"},
     [0]),  # Test case for prediction: <=50K
    ({"age": 44, "workclass": "Private", "fnlgt": 75227, "education": "Prof-school", "education-num": 15,
      "marital-status": "Never-married", "occupation": "Prof-specialty", "relationship": "Not-in-family",
      "race": "White", "sex": "Male", "capital-gain": 14084, "capital-loss": 0, "hours-per-week": 40,
      "native-country": "United-States"},
     [1]),  # Test case for prediction: >50K
    ({"age": 62, "workclass": "Private", "fnlgt": 664366, "education": "Bachelors", "education-num": 13,
      "marital-status": "Married-civ-spouse", "occupation": "Sales", "relationship": "Husband", "race": "White",
      "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 60, "native-country": "United-States"},
     [0]),  # Test case for prediction: <=50K
]

invalid_test_cases = [
    ({"age": "invalid", "workclass": "Private", "fnlgt": 484024, "education": "HS-grad", "education-num": 9,
      "marital-status": "Divorced", "occupation": "Machine-op-inspct", "relationship": "Unmarried", "race": "White",
      "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}),
    ({"age": 36, "workclass": "Private", "fnlgt": 484024, "education": 50, "education-num": 9,
      "marital-status": "Divorced", "occupation": "Machine-op-inspct", "relationship": "Unmarried", "race": "White",
      "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}),
    ({"age": 36, "workclass": "Private", "fnlgt": 484024, "education": "HS-grad", "education-num": 9,
      "marital-status": "Divorced", "occupation": "Machine-op-inspct", "relationship": "Unmarried", "race": "White",
      "sex": "Male"}),
]


@pytest.mark.parametrize("input_data, expected_prediction", valid_test_cases)
def test_predict(input_data, expected_prediction):
    response = test_client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()['prediction'] == expected_prediction
    assert "input" in response.json()
    assert type(response.json()['input']) == dict
    assert type(response.json()['prediction']) == list


@pytest.mark.parametrize("input_data", invalid_test_cases)
def test_predict_invalid_data(input_data):
    response = test_client.post("/predict", json=input_data)
    assert response.status_code == 422


def test_predict_raises_exception():
    error_msg = "example error"
    with patch('main.inference', side_effect=Exception(error_msg)):
        response = test_client.post("/predict", json=valid_test_cases[0][0])
        assert response.status_code == 400
        assert response.json() == {"detail": error_msg}
