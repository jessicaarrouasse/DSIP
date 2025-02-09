from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, C=1.0, class_weight=None,
                 penalty='l2', solver='liblinear', max_iter=1000):
        self.threshold = threshold
        self.C = C
        self.class_weight = class_weight
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        # Create LogisticRegression with the parameters
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter
        )
        self.model.fit(X, y)
        # Add LogisticRegression attributes
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        y_pred_proba = self.model.predict_proba(X)
        return (y_pred_proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)