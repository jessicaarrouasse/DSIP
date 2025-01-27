import argparse
import os
import pickle

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from constants import KNN, LOGISTIC_REGRESSION
from utils import get_data


def _get_data(path: str):
    return get_data(f"{path}/X_train.csv"), get_data(f"{path}/y_train.csv")


def save_model(model_name, grid_search, features_name):
    os.makedirs("models", exist_ok=True)
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)

    # Save best parameters as CSV
    params_df = pd.DataFrame([grid_search.best_params_])
    params_df.to_csv(f'models/{model_name}_best_params.csv', index=False)

    # Save best score as CSV
    score_df = pd.DataFrame([{'best_score': grid_search.best_score_}])
    score_df.to_csv(f'models/{model_name}_best_score.csv', index=False)

    feature_importance = pd.DataFrame({
        'feature': range(features_name),
        'importance': abs(grid_search.best_estimator_.coef_[0])
    })
    feature_importance =feature_importance.sort_values('importance', ascending=False)
    # Save feature importance
    feature_importance.to_csv(f'models/{model_name}_feature_importance.csv', index=False)


def calculate_class_weights(y_train):
    # Calculate class weights
    total_samples = len(y_train)
    class_0 = len(y_train[y_train['is_click'] == 0])
    class_1 = len(y_train[y_train['is_click'] == 1])
    class_weights = {
        0: (total_samples / (2 * class_0)),
        1: (total_samples / (2 * class_1))
    }
    return class_weights


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, C=1.0, class_weight=None,
                 penalty='l2', solver='liblinear', max_iter=1000):
        self.threshold = threshold
        self.C = C
        self.class_weight = class_weight
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        # Create LogisticRegression with the parameters
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter
        )
        self.model.fit(X, y)
        # Add LogisticRegression attributes
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        y_pred_proba = self.model.predict_proba(X)
        return (y_pred_proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def train_logistic_regression(X_train, y_train):
    # model = LogisticRegression()
    model = ThresholdClassifier()
    # Parameter grid for Logistic Regression
    class_weights = calculate_class_weights(y_train)
    param_grid = {
        'threshold': [0.1, 0.2, 0.3, 0.5, 0.6, 0.7],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'class_weight': [class_weights, 'balanced', None],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],  # liblinear supports both l1 and l2
        'max_iter': [1000],  # Increased iterations for convergence
    }

    grid_search = perform_grid_search(X_train, y_train, model, param_grid)
    save_model("logistic_regression", grid_search, X_train.shape[1])
    print("LogisticRegression saved")



def perform_grid_search(X_train, y_train, model, param_grid):
    # Define multiple scoring metrics suitable for imbalanced data
    scoring = {
        'average_precision': make_scorer(average_precision_score),
        'roc_auc': make_scorer(roc_auc_score),
        'recall': make_scorer(recall_score),  # Important for minority class
        'f1': make_scorer(f1_score)
    }

    # Use StratifiedKFold to maintain class distribution in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1',  # Use balanced accuracy to select best model
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    # Fit the grid search
    grid_search.fit(X_train, y_train)

    return grid_search


def train_knn(X_train, y_train, k=None):
    knn = KNeighborsClassifier()
    # param_grid = {
    #     'n_neighbors': [3, 5, 7, 9, 11, 13, 15],  # Different values of k
    #     'weights': ['uniform', 'distance'],  # Weight function used in prediction
    #     'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
    #     'p': [1, 2],  # Power parameter for Minkowski metric
    #     'leaf_size': [10, 20, 30, 40]  # Leaf size for tree data structure
    # }
    param_grid = {
        'n_neighbors': [3],  # Different values of k
        'weights': ['uniform'],  # Weight function used in prediction
        'metric': ['euclidean'],  # Distance metrics
        'p': [1, 2],  # Power parameter for Minkowski metric
    }
    model = perform_grid_search(X_train, y_train, knn, param_grid)
    save_model(model, "knn")
    print("KNN saved")

def unknown_model(*args):
    raise Exception("Model not supported")

def main(path: str, model: str):
    models = {
        KNN: train_knn,
        LOGISTIC_REGRESSION: train_logistic_regression
    }

    X_train, y_train = _get_data(path)

    train_function = models.get(model, unknown_model)
    train_function(X_train, y_train)



if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--models-path", type=str)
   parser.add_argument( "--model", type=str)
   args = parser.parse_args()
   main(args.models_path, args.model)