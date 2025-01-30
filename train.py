import os
import pickle
import logging
import argparse
import wandb
import json
import time
from imblearn.over_sampling import SMOTE
import pandas as pd
from typing import Tuple
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, recall_score, roc_auc_score, accuracy_score, classification_report, roc_curve
from sklearn.tree import DecisionTreeClassifier

# Set up logging configuration
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_pickle(file_path: str) -> object:
    """
    Helper function to load a pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_pickles(data_path: str) -> Tuple:
    """
    Load all the pickled datasets (train, test, validation) from the given data directory.
    """
    logging.info(f"Loading pickles from {data_path}")
    
    # Load the training, testing, and validation data
    X_train, y_train = load_pickle(os.path.join(data_path, "X_train_y_train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "X_test_y_test.pkl"))
    X_validation, y_validation = load_pickle(os.path.join(data_path, "X_validation_y_validation.pkl"))
    
    logging.info("Pickles loaded successfully.")
    
    return X_train, y_train, X_test, y_test, X_validation, y_validation

def load_config(config_path: str):
    """
    Load the configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_model(model, model_name: str):
    """
    Save the trained model to a pickle file.
    """
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {model_path}")

def plot_feature_importances(model, feature_names):
    """
    Plot the feature importances from the trained Random Forest model.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]  # Sort the importances in descending order

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()

from sklearn.utils import resample

def undersample_data(X, y):
    """
    Perform undersampling to balance the dataset by keeping all minority class samples
    and randomly sampling from the majority class.
    """
    # Combine X and y into a single DataFrame for easier manipulation
    df = pd.DataFrame(X)
    df['label'] = y

    # Separate majority and minority classes
    df_minority = df[df['label'] == 1]
    df_majority = df[df['label'] == 0]

    # Downsample majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,  # Sample without replacement
        n_samples=len(df_minority),  # Match the minority class size
        random_state=42  # Ensure reproducibility
    )

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split back into X and y
    y_balanced = df_balanced['label'].values
    X_balanced = df_balanced.drop(columns=['label']).values

    return X_balanced, y_balanced

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    specificity = recall_score(y, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y, y_proba)
    
    logging.info(f"{dataset_name} Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    return {
        f"{dataset_name}_accuracy": accuracy,
        f"{dataset_name}_precision": precision,
        f"{dataset_name}_recall": recall,
        f"{dataset_name}_specificity": specificity,
        f"{dataset_name}_f1_score": f1,
        f"{dataset_name}_roc_auc": roc_auc
    }

def train(data_path: str, model_name: str, config: dict):
    # Create a formatted experiment name
    experiment_name = f"{model_name}_exp_{int(time.time())}"
    # Initialize wandb to track the experiment
    wandb.init(project="ad-click-prediction", name=experiment_name, config=config)
    
    # Load the data
    X_train, y_train, X_test, y_test, X_val, y_val = load_pickles(data_path)
    
    # Combine X_train and X_validation
    # logging.info("Combining training and validation datasets...")
    # X_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], ignore_index=True).values
    # y_train = pd.concat([pd.Series(y_train), pd.Series(y_val)], ignore_index=True).values
    # logging.info(f"Combined training dataset size: {X_train.shape[0]} samples.")
    
    
    # df_train = pd.DataFrame(X_train)
    # df_train['label'] = y_train

    # # Drop rows with null values
    # logging.info("Dropping rows with null values...")
    # df_train = df_train.dropna()
    # logging.info(f"Data size after dropping nulls: {df_train.shape}")

    # # Separate the cleaned data back into X_train and y_train
    # y_train = df_train['label'].values
    # X_train = df_train.drop(columns=['label']).values
    # # Apply SMOTE to the training data
    # logging.info("Applying SMOTE to balance the dataset.")
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    # logging.info(f"Training data size after SMOTE: {X_train.shape[0]} samples.")
    
    # # Apply undersampling to the training data
    # logging.info("Applying undersampling to balance the dataset.")
    # X_train, y_train = undersample_data(X_train, y_train)
    # logging.info(f"Training data size after undersampling: {X_train.shape[0]} samples.")
    
    # # Select features by their indices (you can change this list as needed)
    # features_idx = [0, 1, 2, 3, 4, 5, 6, 8]  # selecting best 8 features
    # X_train = X_train[:, features_idx]
    # X_test = X_test[:, features_idx]
    # X_val = X_val[:, features_idx]
    
    # Initialize and train the Random Forest model using config
    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 100),  # Default value 100
        max_depth=config.get('max_depth', None),        # Default value None
        min_samples_split=config.get('min_samples_split', 2),  # Default value 2
        min_samples_leaf=config.get('min_samples_leaf', 1),    # Default value 1
        class_weight=config.get('class_weight', None),    # Default value None
        random_state=config.get('random_state', 42)      # Default value 42
    )
    # Initialize and train the Decision Tree model using config
    # model = DecisionTreeClassifier(
    #     criterion=config.get('criterion', 'gini'),  # Default is 'gini'
    #     max_depth=config.get('max_depth', None),     # Default value None
    #     min_samples_split=config.get('min_samples_split', 2),  # Default value 2
    #     min_samples_leaf=config.get('min_samples_leaf', 1),    # Default value 1
    #     class_weight=config.get('class_weight', None),    # Default value None
    #     random_state=config.get('random_state', 42)      # Default value 42
    # )
    
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    # Plot feature importances
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]  # don't have feature names
    plot_feature_importances(model, feature_names)
    
    metrics_train = evaluate_model(model, X_train, y_train, "Train")
    metrics_val = evaluate_model(model, X_val, y_val, "Validation")
    metrics_test = evaluate_model(model, X_test, y_test, "Test")
    
    wandb.log({**metrics_train, **metrics_val, **metrics_test})
    
    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {metrics_test['Test_roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title("Test Set ROC Curve")
    plt.show()
    
    # # Evaluate the model on the val set (for now - still finetune)
    # y_val_pred = model.predict(X_test)
    # y_val_proba = model.predict_proba(X_test)[:, 1]
    
    # # Evaluate the model on the test set
    # y_test_pred = model.predict(X_test)
    # y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # # Combine X_val, y_val, and y_pred_val into a DataFrame
    # df_val = pd.DataFrame(X_test)
    # df_val['y_val'] = y_test
    # df_val['y_pred_val'] = y_val_pred
    
    # # Save the DataFrame to a CSV file
    # val_results_path = f"val_results2_{model_name}.csv"
    # df_val.to_csv(val_results_path, index=False)
    # logging.info(f"Validation results saved to {val_results_path}")
    
    # # Calculate metrics
    # accuracy = accuracy_score(y_test, y_val_pred)
    # precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_val_pred, average='binary')
    # specificity = recall_score(y_test, y_val_pred, pos_label=0)
    # roc_auc = roc_auc_score(y_test, y_val_proba)
    
    # # Log metrics to W&B
    # wandb.log({
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall (sensitivity)": recall,
    #     "specificity": specificity,
    #     "f1_score": f1,
    #     "roc_auc": roc_auc
    # })
    
    # logging.info(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # # ROC Curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_val_proba)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    
    # # Save the ROC curve plot
    # plt.savefig(f"roc_curve_{model_name}.png")
    # wandb.log({"roc_curve": wandb.Image(f"roc_curve_{model_name}.png")})
    
    # # Print the classification report and log it to wandb
    # class_report = classification_report(y_test, y_val_pred, output_dict=True)
    # # Convert the report to a DataFrame
    # class_report_df = pd.DataFrame(class_report).transpose()
    # logging.info("Classification Report:\n")
    # logging.info(class_report_df)
    # # Manually add the 'Metrics' column with predefined values
    # class_report_df['Metrics'] = ['0.0', '1.0', 'accuracy', 'macro avg', 'weighted avg']
    # # Convert the DataFrame to a W&B table
    # classification_table = wandb.Table(dataframe=class_report_df)
    # # Log the table to W&B
    # wandb.log({"classification_report": classification_table})
    # logging.info("Classification Report logged to W&B.")
    
    # cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    # plt.savefig(f"confusion_matrix_{model_name}.png")
    # wandb.log({"confusion_matrix": wandb.Image(f"confusion_matrix_{model_name}.png")})
    
    # # Save the trained model
    # save_model(model, model_name)
    
    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the given data")
    parser.add_argument("-d", "--data_path", type=str, help="Path to the data directory")
    parser.add_argument("-m", "--model_name", type=str, help="Name of the model to save")
    parser.add_argument("-c", "--config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Train the model
    train(args.data_path, args.model_name, config)