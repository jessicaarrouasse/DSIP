from enum import Enum

class ModelsName(Enum):
    DecisionTreeClassifier = "sklearn.tree.DecisionTreeClassifier"
    LogisticRegression = "sklearn.linear_model.LogisticRegression"
    RandomForestClassifier = "sklearn.ensemble.RandomForestClassifier"
    GradientBoosting = "sklearn.ensemble.GradientBoostingClassifier"
    SVC = "sklearn.svm.SVC"
    GaussianNB = "sklearn.naive_bayes.GaussianNB"
    MultinomialNB = "sklearn.naive_bayes.MultinomialNB"
    KNeighborsClassifier = "sklearn.neighbors.KNeighborsClassifier"
    # Perceptron = "sklearn.linear_model.Perceptron"
    # SGDClassifier = "sklearn.linear_model.SGDClassifier"
    # XGBClassifier = "xgboost.XGBClassifier"
    # CatBoostClassifier = "catboost.CatBoostClassifier"
    # LGBMClassifier = "lightgbm.LGBMClassifier"
    # LinearDiscriminantAnalysis = "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
    # QuadraticDiscriminantAnalysis = "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"