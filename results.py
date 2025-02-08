import argparse
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import seaborn as sns

from utils import get_data


def compute_precision_recall_curve(y_true, y_scores, model_name: str):
    """
    compute and plot Precision-Recall curve

    our parameters:
        y_true (array): true labels
        y_scores (array): predicted probabilities
        model_name (str): name of the model
    """
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(f'metrics/{model_name}_precision_recall_curve.png')
    plt.close()

    return avg_precision


def compute_roc_curve(y_true, y_scores, model_name: str):
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
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'metrics/{model_name}_roc_curve.png')
    plt.close()

    return roc_auc


def generate_confusion_matrix(y_true, y_pred, model_name):
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
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'metrics/{model_name}_confusion_matrix.png')
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
        'Sensitivity (Recall)': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'Precision': tp / (tp + fp),
        'F1 Score': 2 * tp / (2 * tp + fp + fn),
        'ROC AUC': auc(roc_curve(y_true, y_scores)[0], roc_curve(y_true, y_scores)[1])
    }

    return metrics


def results(predictions_path: str, predictions_proba_path:str,  ground_truth_path: str):
    predictions = get_data(predictions_path, header=False)
    predictions_proba = get_data(predictions_proba_path, header=False)
    ground_truth = get_data(ground_truth_path)

    # Olga: Extract true labels and predictions
    y_true = ground_truth
    y_pred = predictions
    y_scores = predictions_proba[1]
    model_name = predictions_path.split("/")[-1].split("_")[0]

    os.makedirs("metrics", exist_ok=True)
    # Compute ROC curve
    roc_auc = compute_roc_curve(y_true, y_scores, model_name) #Olga: call compute_roc_curve method

    # Generate Precision-Recall Curve and Average Precision
    avg_precision = compute_precision_recall_curve(y_true, y_scores, model_name)

    # Generate confusion matrix
    generate_confusion_matrix(y_true, y_pred, model_name) # Olga: call generate_confusion_matrix method

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores) # Olga: call calculate_metrics method

    # Add Average Precision to metrics
    metrics['Average Precision (AP)'] = avg_precision

    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valupe'])
    metrics_df.to_csv(f'metrics/{model_name}_metrics.csv')

    # Print metrics
    print(f"Metrics for {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Trainer")
   parser.add_argument("-pp", "--predictions-path")
   parser.add_argument("-ppp", "--predictions-proba-path")
   parser.add_argument("-gt", "--ground-truth-path")

   args = parser.parse_args()
   results(args.predictions_path, args.predictions_proba_path, args.ground_truth_path)
