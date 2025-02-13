import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_optimal_threshold(model, X_val, y_val, metric='f1', thresholds=None):
    """
    Find the optimal classification threshold for XGBoost model predictions.

    Parameters:
    -----------
    model : XGBoost model or SklearnWrapper
        The trained XGBoost model
    X_val : array-like or DataFrame
        Validation features
    y_val : array-like
        True labels for validation data
    metric : str, optional (default='f1')
        Metric to optimize for. Options: 'f1', 'precision', 'recall'
    thresholds : array-like, optional
        Custom threshold values to try. If None, generates sequence from 0.1 to 0.9

    Returns:
    --------
    dict
        Contains optimal threshold, best score, and all results for plotting
    """
    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_val)[:, 1]
    else:
        y_scores = model.predict(X_val)

    # Generate thresholds if not provided
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    # Initialize results storage
    results = {
        'thresholds': [],
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': []
    }

    # Try each threshold
    for threshold in thresholds:
        # Convert probabilities to predictions using threshold
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate metrics
        results['thresholds'].append(threshold)
        results['f1_scores'].append(f1_score(y_val, y_pred))
        results['precision_scores'].append(precision_score(y_val, y_pred))
        results['recall_scores'].append(recall_score(y_val, y_pred))

    # Find optimal threshold based on specified metric
    metric_scores = results[f'{metric}_scores']
    optimal_idx = np.argmax(metric_scores)
    optimal_threshold = results['thresholds'][optimal_idx]
    best_score = metric_scores[optimal_idx]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['thresholds'], results['f1_scores'], label='F1 Score')
    plt.plot(results['thresholds'], results['precision_scores'], label='Precision')
    plt.plot(results['thresholds'], results['recall_scores'], label='Recall')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--',
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Scores vs. Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_optimization.png')
    plt.close()

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.close()

    return {
        'optimal_threshold': optimal_threshold,
        'best_score': best_score,
        'results': results
    }


def apply_optimal_threshold(model, optimal_threshold):
    """
    Create a new model wrapper that applies the optimal threshold.

    Parameters:
    -----------
    model : XGBoost model or SklearnWrapper
        The original trained model
    optimal_threshold : float
        The threshold to apply

    Returns:
    --------
    SklearnWrapper
        A modified version of the model that uses the optimal threshold
    """

    class OptimizedThresholdWrapper:
        def __init__(self, base_model, threshold):
            self.base_model = base_model
            self.threshold = threshold

        def predict(self, X):
            if hasattr(self.base_model, 'predict_proba'):
                probs = self.base_model.predict_proba(X)[:, 1]
            else:
                probs = self.base_model.predict(X)
            return (probs >= self.threshold).astype(int)

        def predict_proba(self, X):
            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(X)
            probs = self.base_model.predict(X)
            return np.column_stack((1 - probs, probs))

        def score(self, X, y):
            preds = self.predict(X)
            return f1_score(y, preds)

    return OptimizedThresholdWrapper(model, optimal_threshold)