import argparse
import os
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from constants import LOGISTIC_REGRESSION, ADABOOST, RANDOM_FOREST
from utils import get_data, ThresholdClassifier
from imblearn.under_sampling import NearMiss

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
        'importance': abs(get_feature_importance(model_name, grid_search))
    })
    feature_importance =feature_importance.sort_values('importance', ascending=False)
    # Save feature importance
    feature_importance.to_csv(f'models/{model_name}_feature_importance.csv', index=False)

def get_feature_importance(model_name, grid_search):
    if model_name == ADABOOST or model_name == RANDOM_FOREST:
        return grid_search.best_estimator_.feature_importances_
    if model_name == LOGISTIC_REGRESSION:
        return grid_search.best_estimator_.coef_[0]

def calculate_class_weights(y_train):
    # Calculate class weights based on class ratio
    class_0 = len(y_train[y_train['is_click'] == 0])
    class_1 = len(y_train[y_train['is_click'] == 1])

    class_weights = {
        0: 1,  # majority class weight stays at 1
        1: class_0 / class_1  # minority class weight based on ratio
    }

    return class_weights

def data_resample(X_train, y_train):
    minority_class_count = len(y_train[y_train['is_click'] == 1])
    # Initialize NearMiss
    nm = NearMiss(
        version=1,
        n_neighbors=15,
        sampling_strategy={0: minority_class_count*7}
    )
    X_resampled, y_resampled = nm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

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
    save_model(LOGISTIC_REGRESSION, grid_search, X_train.shape[1])
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
        refit='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    # Fit the grid search
    grid_search.fit(X_train, y_train)

    return grid_search

def train_adaboost(X_train, y_train):
    # Create base estimators with different depths
    dt_1 = DecisionTreeClassifier(max_depth=1, class_weight='balanced')
    dt_2 = DecisionTreeClassifier(max_depth=2, class_weight='balanced')
    dt_3 = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
    AdaBoost= AdaBoostClassifier(random_state=42)

    param_grid = {
        'estimator': [dt_1, dt_2, dt_3],
        'n_estimators': [50, 100, 150, 200, 250, 300, 500, 700] ,
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }

    grid_search = perform_grid_search(X_train, y_train, AdaBoost, param_grid)
    save_model(ADABOOST, grid_search, X_train.shape[1])
    print("AdaBoost saved")

def train_random_forest(X_train, y_train):
    # Initialize the classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        "min_samples_split": [2],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"]
    }
    
    # Perform Grid Search
    grid_search = perform_grid_search(X_train, y_train, rf, param_grid)
    
    # Save the trained model
    save_model(RANDOM_FOREST, grid_search, X_train.shape[1])
    print("RandomForest saved")

def unknown_model(*args):
    raise Exception("Model not supported")

def main(path: str, model: str):
    models = {
        LOGISTIC_REGRESSION: train_logistic_regression,
        ADABOOST: train_adaboost,
        RANDOM_FOREST: train_random_forest
    }

    X_train, y_train = _get_data(path)
    # X_train, y_train = data_resample(X_train, y_train)
    train_function = models.get(model, unknown_model)
    train_function(X_train, y_train)



if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--models-path", type=str)
   parser.add_argument( "--model", type=str)
   args = parser.parse_args()
   main(args.models_path, args.model)