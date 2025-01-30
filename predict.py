import argparse
import os
import pickle
import numpy as np
from results import compute_roc_curve, generate_confusion_matrix, calculate_metrics
import pandas as pd
from constants import NAIVE_BAYES, XGBOOST, LIGHTGBM
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score


class SklearnWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        # If X is already a DMatrix, use it directly
        if isinstance(X, xgb.DMatrix):
            preds = self.model.predict(X)
        else:
            # If it's a DataFrame, use its feature names
            if isinstance(X, pd.DataFrame):
                dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
            # If it's a numpy array, generate feature names
            elif isinstance(X, np.ndarray):
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                dtest = xgb.DMatrix(X, feature_names=feature_names)
            else:
                raise TypeError(f"Unsupported input type: {type(X)}")
            preds = self.model.predict(dtest)

        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        # If X is already a DMatrix, use it directly
        if isinstance(X, xgb.DMatrix):
            preds = self.model.predict(X)
        else:
            # If it's a DataFrame, use its feature names
            if isinstance(X, pd.DataFrame):
                dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
            # If it's a numpy array, generate feature names
            elif isinstance(X, np.ndarray):
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                dtest = xgb.DMatrix(X, feature_names=feature_names)
            else:
                raise TypeError(f"Unsupported input type: {type(X)}")
            preds = self.model.predict(dtest)

        return np.column_stack((1 - preds, preds))

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(y, preds)

def get_data(gender):
    """Load pre-split test data from gender-specific pickle files."""
    file_path = f'data/X_test_y_test_{gender}.pkl'

    with open(file_path, 'rb') as f:
        X_test, y_test = pickle.load(f)

    # Convert numpy array to DataFrame with feature names if necessary
    if isinstance(X_test, np.ndarray):
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test, columns=feature_names)

    return X_test, y_test

def load_model(model_name, gender):
    """Load gender-specific model from disk."""
    with open(f'models/{model_name}_{gender}.pkl', 'rb') as f:
        model = pickle.load(f)

    if model_name == XGBOOST:
        model = SklearnWrapper(model)
    return model


def predict_for_gender(model_name, gender):
    """Make predictions for a specific gender."""
    X_test, y_test = get_data(gender)
    model = load_model(model_name, gender)

    print(f"\nPredicting for {gender} data using {model_name}")
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_scores)

    # Create and save predictions DataFrame
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'predicted_probability': y_scores,
        'gender': gender
    })

    return predictions_df, metrics


def predict(model_name):
    """Main prediction function that handles predictions for both genders."""
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)

    # Get predictions for both genders
    male_predictions, male_metrics = predict_for_gender(model_name, 'male')
    female_predictions, female_metrics = predict_for_gender(model_name, 'female')

    # Combine predictions
    all_predictions = pd.concat([male_predictions, female_predictions], ignore_index=True)

    # Save combined predictions
    predictions_path = f'predictions/{model_name}_predictions.csv'
    all_predictions.to_csv(predictions_path, index=False)

    # Print metrics for both genders
    print(f"\nMetrics for {model_name} (Male):")
    for metric, value in male_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nMetrics for {model_name} (Female):")
    for metric, value in female_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"ROC curves and confusion matrices saved with gender-specific suffixes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Predictor")
    parser.add_argument(
        "-m", "--model_name",
        choices=[NAIVE_BAYES, XGBOOST, LIGHTGBM],
        default=NAIVE_BAYES,
        help="Specify the model to use for prediction"
    )
    parser.add_argument("-e", "--csv_path", default=NAIVE_BAYES)

    args = parser.parse_args()
    predict(args.model_name)
