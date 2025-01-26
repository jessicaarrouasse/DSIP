grid_search_params = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10, 100],  # Strength of regularization
        "penalty": ["l1", "l2", "elasticnet", "none"],  # Type of regularization
        "solver": ["liblinear", "lbfgs", "saga"],  # Solver used in the optimization
        "max_iter": [100, 200, 500],  # Maximum number of iterations
    },
    "RandomForestClassifier": {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
        # "n_estimators": [10, 50, 100, 200],  # Number of trees
        # "max_depth": [None, 10, 20, 30],  # Maximum depth of the trees
        # "min_samples_split": [2, 5, 10],  # Minimum samples required to split a node
        # "min_samples_leaf": [1, 2, 4],  # Minimum samples required at a leaf node
        # "bootstrap": [True, False],  # Whether bootstrap samples are used
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
        "kernel": ["linear", "poly", "rbf", "sigmoid"],  # Kernel type
        "gamma": ["scale", "auto"],  # Kernel coefficient
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9],  # Number of neighbors
        "weights": ["uniform", "distance"],  # Weight function used in prediction
        "metric": ["euclidean", "manhattan", "minkowski"],  # Distance metric
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100, 200],  # Number of boosting stages
        "learning_rate": [0.01, 0.1, 0.2],  # Learning rate
        "max_depth": [3, 5, 10],  # Maximum depth of the individual estimators
        "min_samples_split": [2, 5, 10],  # Minimum samples required to split a node
        "min_samples_leaf": [1, 2, 4],  # Minimum samples required at a leaf node
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200],  # Number of trees
        "learning_rate": [0.01, 0.1, 0.2],  # Learning rate
        "max_depth": [3, 5, 10],  # Maximum tree depth
        "subsample": [0.8, 1.0],  # Subsample ratio
        "colsample_bytree": [0.8, 1.0],  # Ratio of columns sampled per tree
    }
}
