import json
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : LogisticRegression
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def save_model_pickle(model, output_path=os.path.join('model', 'model.pkl')):
    """
    Save input model as pkl file.

    Inputs
    ------
    model
        Model to be saved.
    output_path : str
        Path to save the pkl file.
    """
    with open(output_path, 'wb') as model_file:
        pickle.dump(model, model_file)


def compute_model_metrics(y, preds, log=True):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    log : bool
        Whether to log metrics or not.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    if log:
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Fbeta: {fbeta}")
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs:
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred


def evaluate_slice(model, data, cat_features, encoder, lb, output_path=os.path.join('model', 'slice_evaluation.json')):
    """Evaluate the model on slices of dataset.

    Inputs:
    -------
    model : LogisticRegression
        Trained machine learning model.
    data : pd.DataFrame
        Pandas DataFrame containing the dataset.
    cat_features : list
        List of categorical feature names.
    output_path : str
        Path to save the evaluation results in JSON format. Defaults to 'model/slice_evaluation.json'.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer passed in.
    """

    slice_evaluation = []

    for feature in cat_features:
        for category in data[feature].unique():
            slice = data[data[feature] == category]

            X_test, y_test, _, _ = (
                process_data(slice, categorical_features=cat_features, label='salary', training=False,
                             encoder=encoder, lb=lb))

            y_pred = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred, log=False)
            slice_evaluation.append(
                {'feature': feature, 'category': category, 'precision': precision, 'recall': recall, 'fbeta': fbeta})

    print(slice_evaluation)
    with open(output_path, 'w') as f:
        json.dump(slice_evaluation, f, indent=4)
