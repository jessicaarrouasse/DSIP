import argparse
import os
import pickle
import pandas as pd
import numpy as np
import argparse
import importlib
import inspect
from joblib import dump
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from models_name import ModelsName


def get_data():
    with open("data/X_train_y_train.pkl", "rb") as train_file:
        X_train, y_train = pickle.load(train_file)
    with open("data/X_test_y_test.pkl", "rb") as test_file:
        X_test, y_test = pickle.load(test_file)
    with open("data/X_validation_y_validation.pkl", "rb") as validation_file:
        X_validation, y_validation = pickle.load(validation_file)

    # Convert Numpy Arrays to Pandas objects
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_validation = pd.DataFrame(X_validation)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    y_validation = pd.Series(y_validation)

    return X_train, y_train, X_test, y_test, X_validation, y_validation


def save_model(model, model_name):
    # save
    os.makedirs("models", exist_ok=True)

    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)


def get_model_dynamically(model_name, **model_params):
    """
    Dynamically load a model from sklearn and initialize it with user-defined parameters.

    Args:
        model_name (str): Name of the sklearn model (e.g., "LinearRegression").
        **model_params: Additional keyword arguments for the model initialization.

    Returns:
        sklearn model instance.
    """
    try:
        # Match the model name to the corresponding module
        module_path = ModelsName[model_name].value

        # Split module path and class name
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        
       # Filter valid parameters for the model
        model_signature = inspect.signature(model_class)
        valid_params = {
            param: value
            for param, value in model_params.items()
            if param in model_signature.parameters
        }

        # Initialize the model
        return model_class(**valid_params)
    except KeyError:
        raise ValueError(f"Model '{model_name}' is not defined in ModelsName Enum.")
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Failed to load model '{model_name}' from '{module_path}'.") from e


def cross_validate_model(model_name, model_params, X, y, n_splits=5):
    """
    Perform K-Fold Cross-Validation on a model.

    Args:
        model_name (str): Name of the model.
        model_params (dict): Parameters to pass to the model.
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable.
        n_splits (int): Number of folds for K-Fold Cross-Validation.

    Returns:
        tuple: The last trained model and the average metrics (dict).
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mse_scores, accuracy_scores, auc_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], [], []
    fold = 1

    for train_idx, test_idx in kfold.split(X, y):
        # Split data into train and validation folds
        X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize and train the model
        model = get_model_dynamically(model_name, **model_params)
        model.fit(X_fold_train, y_fold_train)

        # Predict on the validation fold
        y_pred = model.predict(X_fold_test)
        y_pred_proba = model.predict_proba(X_fold_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Evaluate metrics for this fold
        mse_scores.append(mean_squared_error(y_fold_test, y_pred))
        accuracy_scores.append(accuracy_score(y_fold_test, y_pred.round()))
        auc_scores.append(roc_auc_score(y_fold_test, y_pred))
        precision_scores.append(precision_score(y_fold_test, y_pred.round()))
        recall_scores.append(recall_score(y_fold_test, y_pred.round()))
        f1_scores.append(f1_score(y_fold_test, y_pred.round()))

        print(f"Fold {fold}: MSE = {mse_scores[-1]:.4f}, Accuracy = {accuracy_scores[-1]:.4f}")
        fold += 1

    # Average metrics across folds
    metrics = {
        "mse": sum(mse_scores) / len(mse_scores),
        "accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "auc": sum(auc_scores) / len(auc_scores),
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }

    print("\nCross-Validation Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return model, metrics


def train(model_name, model_params, split_num=5):
    """
    Train and evaluate a model, and save predictions plots to a PDF.

    Args:
        model_name (str): Name of the model.
        model_params (dict): Parameters to pass to the model.
        split_num (int): Current split number.

    Returns:
        tuple: The trained model and the Mean Squared Error (MSE).
    """

    print('model_params:', model_params)
    
    X_train, y_train, X_test, y_test, X_validation, y_validation = get_data()


    if split_num > 1:
        # Combine train and test sets for cross-validation
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)

        # Perform Cross-Validation
        model, metrics = cross_validate_model(model_name, model_params, X_combined, y_combined)
        return model, metrics["mse"]
    else:
        # Regular training process
        model = get_model_dynamically(model_name, **model_params)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Evaluate the model on the test set
        mse = mean_squared_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred.round())
        auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred.round())
        recall = recall_score(y_test, y_pred.round())
        f1 = f1_score(y_test, y_pred.round())

        print(f"{model_name} - Split {split_num}")
        print(f"Mean Squared Error: {mse}")
        print(f"Accuracy: {accuracy}")
        print(f"AUC-ROC: {auc}")
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        return model, mse



def evaluate_model(y_test, y_pred, y_pred_proba=None, model_name=None, split_num=None):
    """
    Evaluate the model's performance and print key metrics.

    Parameters:
        y_test (array-like): True labels of the test set.
        y_pred (array-like): Predicted labels of the test set.
        y_pred_proba (array-like, optional): Predicted probabilities (if available).
        model_name (str, optional): Name of the model being evaluated.
        split_num (int, optional): Split number (for tracking during cross-validation).

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred.round())
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    precision = precision_score(y_test, y_pred.round())
    recall = recall_score(y_test, y_pred.round())
    f1 = f1_score(y_test, y_pred.round())

    if model_name and split_num is not None:
        print(f"{model_name} - Split {split_num}")
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy: {accuracy}")
    if auc is not None:
        print(f"AUC-ROC: {auc}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    return {
        'mse': mse,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument('-m', '--model-name')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-ms', '--max-steps', type=int)
    parser.add_argument('-bs', '--batch-size', type=int)
    parser.add_argument('-lr', '--learning-rate', type=float, help="Learning rate")
    parser.add_argument('--alpha', type=float, help="Alpha parameter for Ridge/Lasso (if applicable)")
    parser.add_argument('--max-depth', type=int, help="Max depth for tree-based models (if applicable)")
    parser.add_argument('--max-iter', type=int, help="Maximum number of iterations for the model.")
    parser.add_argument('--n-estimators', type=int, help="Number of trees in RandomForest.")
    parser.add_argument('--kernel', type=str,  help="Kernel type for SVC.")
    parser.add_argument('--C', type=float, help="Regularization parameter for SVC.")
    parser.add_argument('--gamma', type=str, help="Gamma parameter for SVC.")
    parser.add_argument('--random-state', type=int, help="Random state for reproducibility.")


    args = parser.parse_args()

    # Prepare model parameters
    model_params = {
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "max_iter": args.max_iter,
        "class_weight": "balanced",
        "n_estimators": args.n_estimators,
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate  # Not all models support learning_rate
    }

    # Remove None values (only pass valid parameters)
    model_params = {k: v for k, v in model_params.items() if v is not None}
    

    train(model_name=args.model_name, model_params=model_params)

