import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def compute_roc_curve(y_true, y_scores, model_name):
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
    plt.savefig(f'{model_name}_roc_curve.png')
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
    plt.savefig(f'{model_name}_confusion_matrix.png')
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


def results():
    predictions = get_data("predictions.csv")

    # Olga: Extract true labels and predictions
    y_true = predictions_df['true_label']
    y_pred = predictions_df['predicted_label']
    y_scores = predictions_df['predicted_probability']

    # Compute ROC curve
    roc_auc = compute_roc_curve(y_true, y_scores, model_name) #Olga: call compute_roc_curve method

    # Generate confusion matrix
    generate_confusion_matrix(y_true, y_pred, model_name) # Olga: call generate_confusion_matrix method

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores) # Olga: call calculate_metrics method

    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.to_csv(f'{model_name}_metrics.csv')

    # Print metrics
    print(f"Metrics for {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    #compute_roc_curve(predictions) # Jessica's line - I just commented this



if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Trainer")
   parser.add_argument("-m", "--model_name", default=DECISION_TREE)
   parser.add_argument("-e", "--csv_path", default=DECISION_TREE)

   args = parser.parse_args()
   results(args.model_name, args.csv_path)
