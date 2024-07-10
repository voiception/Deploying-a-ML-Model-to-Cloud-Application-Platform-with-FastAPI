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
    assert response.json() == {"message": "Hello! Welcome to ML Cloud API"}


valid_test_cases = [
    ({"age": 28, "workclass": "Self-emp-not-inc", "fnlgt": 204519, "education": "Masters", "education-num": 14,
      "marital-status": "Married-spouse-absent", "occupation": "Exec-managerial", "relationship": "Wife", "race": "Asian-Pac-Islander",
      "sex": "Female", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50, "native-country": "India"},
     [0]),

    ({"age": 37, "workclass": "Local-gov", "fnlgt": 149184, "education": "Doctorate", "education-num": 16,
      "marital-status": "Separated", "occupation": "Prof-specialty", "relationship": "Not-in-family",
      "race": "Black", "sex": "Male", "capital-gain": 7688, "capital-loss": 0, "hours-per-week": 38,
      "native-country": "Canada"},
     [1]),

    ({"age": 53, "workclass": "State-gov", "fnlgt": 204018, "education": "Assoc-acdm", "education-num": 12,
      "marital-status": "Widowed", "occupation": "Tech-support", "relationship": "Own-child", "race": "Other",
      "sex": "Female", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 45, "native-country": "Mexico"},
     [0]),
]


invalid_test_cases = [
    ({"age": "thirty-two", "workclass": "Private", "fnlgt": 305920, "education": "Bachelors", "education-num": 13,
      "marital-status": "Separated", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "Black",
      "sex": "Female", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "Philippines"}),

    ({"age": 44, "workclass": "Private", "fnlgt": 185173, "education": True, "education-num": 10,
      "marital-status": "Married-civ-spouse", "occupation": "Sales", "relationship": "Husband", "race": "White",
      "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 35, "native-country": "Germany"}),

    ({"age": 29, "workclass": "Federal-gov", "fnlgt": 98293, "education": "Some-college", "education-num": 10,
      "marital-status": "Married-AF-spouse", "occupation": "Handlers-cleaners", "relationship": "Husband", "race": "White",
      "sex": "Male"})
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
