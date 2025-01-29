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

# Add this class at the top of the file, after imports
class SklearnWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return np.column_stack((1 - preds, preds))

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(y, preds)

def get_data():
    #df = pd.read_csv(csv_path)
    #return df

    """Load pre-split test data from gender-specific pickle files."""
    if gender:
        file_path = f'data/X_test_y_test_{gender}.pkl'
    else:
        file_path = 'data/X_test_y_test.pkl'

    with open(file_path, 'rb') as f:
        X_test, y_test = pickle.load(f)

    return X_test, y_test

def load_model(model_name):
    # load
    with open(f'models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model_name):
    #predict_df = get_data(csv_path) # are these data without labels?
    X_test, y_test = get_data()
    model = load_model(model_name)
    #predictions = model.predict(predict_df)

    print("Predict the model")
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_scores)

    # Create and save predictions DataFrame
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'predicted_probability': y_scores
    })

    # Save predictions
    os.makedirs('predictions', exist_ok=True)
    predictions_df.to_csv(f'predictions/{model_name}_predictions.csv', index=False)

    # Print metrics
    print(f"\nMetrics for {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nPredictions saved to: predictions/{model_name}_predictions.csv")
    print(f"ROC curve saved as: {model_name}_roc_curve.png")
    print(f"Confusion matrix saved as: {model_name}_confusion_matrix.png")

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
