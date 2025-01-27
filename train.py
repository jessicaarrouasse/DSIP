import os
import pickle
import logging
import argparse
import wandb
import pandas as pd
from typing import Tuple
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, recall_score, roc_auc_score, accuracy_score, classification_report, roc_curve


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

def save_model(model, model_name: str):
    """
    Save the trained model to a pickle file.
    """
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {model_path}")

def train(data_path: str, model_name: str):
    # Initialize wandb to track the experiment
    wandb.init(project="ad-click-prediction", name=model_name)
    
    # Load the data
    X_train, y_train, X_test, y_test, X_val, y_val = load_pickles(data_path)
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    
    # Evaluate the model on the val set (for now - still finetune)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')
    specificity = recall_score(y_val, y_val_pred, pos_label=0)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc
    })
    
    logging.info(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save the ROC curve plot
    plt.savefig(f"roc_curve_{model_name}.png")
    wandb.log({"roc_curve": wandb.Image(f"roc_curve_{model_name}.png")})
    
    # Print the classification report and log it to wandb
    class_report = classification_report(y_val, y_val_pred, output_dict=True)
    # Convert the report to a DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()
    logging.info("Classification Report:\n")
    logging.info(class_report_df)
    # Manually add the 'Metrics' column with predefined values
    class_report_df['Metrics'] = ['0.0', '1.0', 'accuracy', 'macro avg', 'weighted avg']
    # Convert the DataFrame to a W&B table
    classification_table = wandb.Table(dataframe=class_report_df)
    # Log the table to W&B
    wandb.log({"classification_report": classification_table})
    logging.info("Classification Report logged to W&B.")
    
    cm = ConfusionMatrixDisplay.from_estimator(model, X_val, y_val)
    plt.savefig(f"confusion_matrix_{model_name}.png")
    wandb.log({"confusion_matrix": wandb.Image(f"confusion_matrix_{model_name}.png")})
    
    # Save the trained model
    save_model(model, model_name)
    
    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the given data")
    parser.add_argument("-d", "--data_path", type=str, help="Path to the data directory")
    parser.add_argument("-m", "--model_name", type=str, help="Name of the model to save")
    args = parser.parse_args()
    
    # Train the model
    train(args.data_path, args.model_name)