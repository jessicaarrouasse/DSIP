import argparse
import os
import pickle
import pandas as pd
import numpy as np
import importlib
import inspect
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import KFold, GridSearchCV
from models_name import ModelsName
import wandb
from dotenv import load_dotenv
import os
from grid_search_params import grid_search_params
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load data function
def get_data():
    with open("data/X_train_y_train.pkl", "rb") as train_file:
        X_train, y_train = pickle.load(train_file)
    with open("data/X_test_y_test.pkl", "rb") as test_file:
        X_test, y_test = pickle.load(test_file)
    
    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    y_train, y_test = pd.Series(y_train), pd.Series(y_test)


    return X_train, y_train, X_test, y_test

# Save model function
def save_model(model, model_name):
    # os.makedirs("models", exist_ok=True)
    # path = f'models/{model_name}.pkl'
    # with open(path, 'wb') as f:
    #     pickle.dump(model, f)
    # wandb.save(path)
    pass

# Load model dynamically
def get_model_dynamically(model_name, **model_params):
    try:
        module_path = ModelsName[model_name].value
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        valid_params = {
            param: value
            for param, value in model_params.items()
            if param in inspect.signature(model_class).parameters
        }

        return model_class(**valid_params)
    except Exception as e:
        raise ValueError(f"Error loading model '{model_name}': {e}")

# Perform grid search
def perform_grid_search(model_name, model_params, X, y, grid_params, scoring="accuracy", cv=3):
    model = get_model_dynamically(model_name, **model_params)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_params,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    grid_search.fit(X, y)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred.round()),
        'precision': precision_score(y_true, y_pred.round()),
        'recall': recall_score(y_true, y_pred.round()),
        'f1': f1_score(y_true, y_pred.round()),
        'auc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    }

# def log_to_wandb(metrics, prefix=None):
#     print('metrics', metrics)

#     tables = {} 

#     for key, value in metrics.items():
#         full_key = f"{prefix}_{key}" if prefix else key
#         parts = full_key.split("_")

#         if len(parts) >= 3:
#             fold = parts[1]
#             metric_name = "_".join(parts[2:])
#         elif len(parts) == 2:
#             fold = "avg"
#             metric_name = parts[1]
#         else:
#             print(f"Skipping invalid metric key: {full_key}")
#             continue

#         if fold not in tables:
#             tables[fold] = wandb.Table(columns=["Metric", "Value"])

#         tables[fold].add_data(metric_name, value)

#     for fold, table in tables.items():
#         wandb.log({f"Metrics Table - Fold {fold}": table})

def log_classification_report(y_true, y_pred, fold):
    class_report = classification_report(y_true, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    
    class_report_df.insert(0, "Metric", class_report_df.index)  
    
    classification_table = wandb.Table(dataframe=class_report_df)
    wandb.log({f"Classification_Report_Fold_{fold}": classification_table})



# Cross-validation
def cross_validate_model(model_name, model_params, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = get_model_dynamically(model_name, **model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, model_name, fold)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        if y_pred_proba is not None:
            plot_roc_curve(y_test, y_pred_proba, model_name, fold)

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        # log_to_wandb(metrics, prefix=f"fold_{fold}")
        log_classification_report(y_test, y_pred, fold)
        fold_metrics.append(metrics)

    avg_metrics = {
        key: np.mean([fold[key] for fold in fold_metrics if fold[key] is not None])
        for key in fold_metrics[0]
    }
    # log_to_wandb(avg_metrics, prefix="avg")


    return avg_metrics

def login_to_wandb(model_name, model_params):

    # Load environment variables
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    wandb.login(key=wandb_api_key)
    wandb.init(
        project="my-model-training",
        name=f"{model_name}_experiment",
        config={
            "model_name": model_name,
            "batch_size": model_params.get("batch_size"),
            "learning_rate": model_params.get("learning_rate"),
            "alpha": model_params.get("alpha"),
            "max_depth": model_params.get("max_depth"),
            "n_estimators": model_params.get("n_estimators"),
            "cv_folds": model_params.get("cv")
        }
    )

# Train function
def train(model_name, model_params, split_num=5, do_grid_search=False):
    X_train, y_train, X_test, y_test = get_data()
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    if do_grid_search:
        print("Performing grid search...")
        grid_params = grid_search_params[model_name]
        model, model_params = perform_grid_search(model_name, model_params, X_train, y_train, grid_params)
        print(f"Best parameters: {model_params}")
    else:
        model = get_model_dynamically(model_name, **model_params)

    login_to_wandb(model_name, model_params)

    if split_num > 1:
        return cross_validate_model(model_name, model_params, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        # log_to_wandb(metrics, prefix="test")
        log_classification_report(y_test, y_pred, 1)


        save_model(model, model_name)
        return metrics
    
def plot_confusion_matrix(y_true, y_pred, model_name, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name} (Fold {fold})")

    filename = f"confusion_matrix_fold_{fold}.png"
    plt.savefig(filename)
    wandb.log({f"confusion_matrix_fold_{fold}": wandb.Image(filename)})
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, model_name, fold):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="blue")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} (Fold {fold})")
    plt.legend(loc="lower right")

    filename = f"roc_curve_fold_{fold}.png"
    plt.savefig(filename)
    wandb.log({f"roc_curve_fold_{fold}": wandb.Image(filename)})
    plt.close()


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument('-m', '--model-name', required=True, help="Name of the model")
    parser.add_argument('-bs', '--batch-size', type=int, help="Batch size")
    parser.add_argument('-lr', '--learning-rate', type=float, help="Learning rate")
    parser.add_argument('--alpha', type=float, help="Alpha parameter for Ridge/Lasso")
    parser.add_argument('--max-depth', type=int, help="Max depth for tree-based models")
    parser.add_argument('--n-estimators', type=int, help="Number of trees in ensemble methods")
    parser.add_argument('--random-state', type=int, default=42, help="Random state for reproducibility")
    parser.add_argument('--cv', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--grid-search', type=bool, help="Run grid search for hyperparameter tuning")
    parser.add_argument('--grid-params', type=str, help="Path to grid search parameters file")
    parser.add_argument('--kernel', type=str, help="Kernel type for SVC (e.g., 'linear', 'rbf')")
    parser.add_argument('--regularization', type=float, help="Regularization parameter (C) for SVM")
    parser.add_argument('--gamma', type=str, help="Gamma parameter for SVM (e.g., 'scale', 'auto')")
    parser.add_argument('--max-iter', type=int, help="Maximum number of iterations for the model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--max-steps', type=int, help="Maximum number of steps for training")



    args = parser.parse_args()

    model_params = {
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "learning_rate": args.learning_rate,
        "kernel": args.kernel,
        "C": args.regularization,  # Regularization is mapped to `C`
        "gamma": args.gamma,
        "max_iter": args.max_iter,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "class_weight": "balanced"
    }
    model_params = {k: v for k, v in model_params.items() if v is not None}

    train(args.model_name, model_params, split_num=args.cv, do_grid_search=args.grid_search)

    wandb.finish()
