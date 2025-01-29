import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, balanced_accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from constants import NAIVE_BAYES, XGBOOST, LIGHTGBM
import wandb

def get_data(predictions_file):
    """Load predictions from CSV file."""
    return pd.read_csv(f'predictions/{predictions_file}')

def compute_roc_curve(y_true, y_scores, model_name, gender):
    #pass
    """
        Olga: compute and plot ROC curve - still not sure that we need this

        our parameters:
            y_true (array): true labels
            y_scores (array): predicted probabilities
            model_name (str): name of the model
        """
    # ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({gender})')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_{gender}_roc_curve.png')

    if wandb.run is not None:
        wandb.log({f"roc_curve_{gender}": wandb.Image(plt)})

    plt.close()

    return roc_auc

def generate_confusion_matrix(y_true, y_pred, model_name,gender):
    """
    Olga: generate confusion matrix plot

    our parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
        model_name (str): Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name} ({gender})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_{gender}_confusion_matrix.png')

    # Log to WandB
    if wandb.run is not None:
        wandb.log({f"confusion_matrix_{gender}": wandb.Image(plt)})

    plt.close()

def calculate_metrics(y_true, y_pred, y_scores):
    """
    Olga: calculate classification metrics

    our parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_scores (array): Predicted probabilities

    returns:
        dict: Comprehensive metrics
    """

    # confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # metrics
    metrics = {
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'Precision': tp / (tp + fp),
        'F1 Score': 2 * tp / (2 * tp + fp + fn),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'ROC AUC': auc(roc_curve(y_true, y_scores)[0], roc_curve(y_true, y_scores)[1])
    }

    # Log metrics to WandB
    if wandb.run is not None:
        wandb.log(metrics)

    return metrics


def analyze_gender_results(predictions_df, model_name, gender):
    """Analyze results for a specific gender."""
    gender_df = predictions_df[predictions_df['gender'] == gender]

    y_true = gender_df['true_label']
    y_pred = gender_df['predicted_label']
    y_scores = gender_df['predicted_probability']

    # Compute ROC curve
    roc_auc = compute_roc_curve(y_true, y_scores, model_name, gender)

    # Generate confusion matrix
    generate_confusion_matrix(y_true, y_pred, model_name, gender)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)

    # Log gender-specific metrics to WandB
    if wandb.run is not None:
        wandb.log({f"{metric}_{gender}": value for metric, value in metrics.items()})

    return metrics


def results(model_name):
    """Generate results for both gender models."""
    # Initialize WandB run
    wandb.init(
        project="ad-click-prediction",
        name=f"{model_name}_evaluation",
        config={"model_type": model_name}
    )

    # Load predictions
    predictions_df = get_data(f"{model_name}_predictions.csv")

    # Analyze results for each gender
    male_metrics = analyze_gender_results(predictions_df, model_name, 'male')
    female_metrics = analyze_gender_results(predictions_df, model_name, 'female')

    # Calculate and save combined metrics
    combined_metrics = calculate_metrics(
        predictions_df['true_label'],
        predictions_df['predicted_label'],
        predictions_df['predicted_probability']
    )

    # Create comprehensive metrics DataFrame
    metrics_df = pd.DataFrame({
        'Male': male_metrics,
        'Female': female_metrics,
        'Combined': combined_metrics
    })

    # Save metrics to CSV
    metrics_df.to_csv(f'{model_name}_metrics.csv')

    # Print results
    print(f"\nMetrics for {model_name}:")
    print("\nMale Model Metrics:")
    for metric, value in male_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nFemale Model Metrics:")
    for metric, value in female_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nCombined Metrics:")
    for metric, value in combined_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Close WandB run
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Results Generator")
    parser.add_argument(
        "-m", "--model_name",
        choices=[NAIVE_BAYES, XGBOOST, LIGHTGBM],
        default=NAIVE_BAYES,
        help="Specify the model for generating results"
    )
    parser.add_argument("-e", "--csv_path", default=NAIVE_BAYES)

    args = parser.parse_args()
    results(args.model_name)
