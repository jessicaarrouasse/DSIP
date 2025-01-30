import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb
import lightgbm as lgb


class SklearnWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return (preds > 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(y, preds)


def get_data(gender):
    """Load pre-split train and validation data for specific gender from pickle files."""
    with open(f'data/X_train_y_train_{gender}.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(f'data/X_validation_y_validation_{gender}.pkl', 'rb') as f:
        X_val, y_val = pickle.load(f)

    if isinstance(X_train, np.ndarray):
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_val = pd.DataFrame(X_val, columns=feature_names)

    return X_train, y_train, X_val, y_val

def save_model(model, model_name, gender):
    """Save trained model to disk with gender specification."""
    os.makedirs("models", exist_ok=True)
    with open(f'models/{model_name}_{gender}.pkl', 'wb') as f:
        pickle.dump(model, f)


def get_class_weights(y_train):
    """Calculate balanced class weights."""
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]
    return scale_pos_weight

def train_model_for_gender(model_name, gender):
    """Train a specific model for a given gender."""
    # Load gender-specific data
    X_train, y_train, X_val, y_val = get_data(gender)
    print(f"\nTraining {model_name} model for {gender}")
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    # Get feature names for feature importance analysis
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in
                                                                                  range(X_train.shape[1])]

    # Train selected model
    model_trainers = {
        'naive_bayes': lambda: train_naive_bayes(X_train, y_train),
        'xgboost': lambda: train_xgboost(X_train, y_train, X_val, y_val),
        'lightgbm': lambda: train_lightgbm(X_train, y_train, X_val, y_val)
    }

    if model_name not in model_trainers:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_trainers[model_name]()

    # Evaluate on validation set
    val_accuracy = model.score(X_val, y_val)
    print(f"Validation Accuracy for {gender}: {val_accuracy:.4f}")

    # Save the gender-specific model
    save_model(model, model_name, gender)
    return model, val_accuracy


def save_feature_importance(importance_results, model_name, gender):
    """Save feature importance results to disk."""
    os.makedirs("feature_importance", exist_ok=True)

    # Save built-in importance
    if 'built_in' in importance_results:
        importance_results['built_in'].to_csv(
            f'feature_importance/{model_name}_{gender}_built_in_importance.csv',
            index=False
        )

    # Save permutation importance
    if 'permutation' in importance_results:
        importance_results['permutation'].to_csv(
            f'feature_importance/{model_name}_{gender}_permutation_importance.csv',
            index=False
        )

    # Save SHAP values in a numpy format
    if 'shap_values' in importance_results:
        np.save(
            f'feature_importance/{model_name}_{gender}_shap_values.npy',
            importance_results['shap_values']
        )

def train(model_name):
    """Main training function that trains separate models for each gender."""
    results = {}

    # Train for male data
    male_model, male_accuracy = train_model_for_gender(model_name, 'male')
    results['male'] = {
        'model': male_model,
        'accuracy': male_accuracy
    }

    # Train for female data
    female_model, female_accuracy = train_model_for_gender(model_name, 'female')
    results['female'] = {
        'model': female_model,
        'accuracy': female_accuracy
    }

    # Print comparative results
    print("\nComparative Results:")
    print(f"Male Model Validation Accuracy: {results['male']['accuracy']:.4f}")
    print(f"Female Model Validation Accuracy: {results['female']['accuracy']:.4f}")

    return results

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model with GridSearch."""
    param_grid = {
        'var_smoothing': np.logspace(-12, -2, 15)
    }

    model = GaussianNB()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=['balanced_accuracy', 'roc_auc', 'f1'],
        refit='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    print("Training Naive Bayes model...")
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_



def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model with GridSearch."""
    scale_pos_weight = get_class_weights(y_train)

    # Define parameter grid
    param_grid = {
        'max_depth': [3],
        'learning_rate': [0.3],
        'n_estimators': [200],
        'min_child_weight': [20],
        'subsample': [0.9],
        'colsample_bytree': [0.9],
        'scale_pos_weight': [scale_pos_weight]
    }

    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    best_score = float('-inf')
    best_params = None
    best_model = None

    print("Performing grid search for XGBoost...")

    # Manual grid search implementation
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for n_estimators in param_grid['n_estimators']:
                for min_child_weight in param_grid['min_child_weight']:
                    for subsample in param_grid['subsample']:
                        for colsample_bytree in param_grid['colsample_bytree']:
                            params = {
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'n_estimators': n_estimators,
                                'min_child_weight': min_child_weight,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'scale_pos_weight': scale_pos_weight,
                                'objective': 'binary:logistic',
                                'eval_metric': 'logloss',
                                'verbosity': 0,
                                'seed': 42
                            }

                            # Train model with early stopping
                            model = xgb.train(
                                params,
                                dtrain,
                                num_boost_round=n_estimators,
                                evals=[(dtrain, 'train'), (dval, 'val')],
                                early_stopping_rounds=20,
                                verbose_eval=False
                            )

                            # Get validation predictions
                            val_preds = model.predict(dval)
                            val_preds_binary = (val_preds > 0.5).astype(int)

                            # Calculate F1 score
                            current_score = f1_score(y_val, val_preds_binary)

                            if current_score > best_score:
                                best_score = current_score
                                best_params = params
                                best_model = model

    print(f"Best Parameters: {best_params}")
    print(f"Best Validation F1 Score: {best_score:.4f}")

    return SklearnWrapper(best_model)


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model with GridSearch, optimized for performance."""
    scale_pos_weight = get_class_weights(y_train)

    # Reduced parameter grid for faster training
    param_grid = {
        'num_leaves': [31],  # Fixed for faster search
        'learning_rate': [0.1],  # Single higher learning rate for quick convergence
        'n_estimators': [200],  # Balanced between speed and model complexity
        'min_child_samples': [20],  # Standard choice to avoid overfitting
        'subsample': [0.8],  # Common best practice value
        'colsample_bytree': [0.8],  # Balanced option
        'is_unbalance': [True]  # Handle class imbalance efficiently
    }

    # Base model with optimized parameters for speed
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=-1,  # Reduce verbosity
        force_row_wise=True,  # Remove testing overhead
        deterministic=True  # Ensure reproducibility
    )

    # Use fewer folds for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Configure GridSearchCV with optimized parameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(f1_score),
        cv=cv,
        n_jobs=1,  # Since LightGBM is using all cores, keep GridSearchCV single-threaded
        verbose=1,
        error_score='raise'
    )

    print("Starting LightGBM grid search with optimized parameters...")

    try:
        # Fit with early stopping
        grid_search.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(period=20)
            ]
        )

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

        # Train final model with best parameters
        final_model = lgb.LGBMClassifier(**grid_search.best_params_)
        final_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(period=20)
            ]
        )

        return final_model

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Returning best model so far...")
        return grid_search.best_estimator_
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train gender-specific models")
    parser.add_argument(
        "-m", "--model_name",
        choices=['naive_bayes', 'xgboost', 'lightgbm'],
        required=True,
        help="Specify the model to train"
    )
    args = parser.parse_args()
    train(args.model_name)