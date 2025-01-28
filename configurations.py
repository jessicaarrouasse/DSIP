CONFIGURATIONS = {
    "LogisticRegression_Default": {
        "model_name": "LogisticRegression",
        "epochs": 20,
        "max_steps": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "max_iter": 100,
        "random_state": 42,
    },
    "RandomForest_Basic": {
        "model_name": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    },
    "SVC_RBF": {
        "model_name": "SVC",
        "kernel": "rbf",
        "regularization": 1.0,
        "gamma": "scale",
        "max_iter": 1000,
        "random_state": 42,
    },
    "SVC_POLY": {
        "model_name": "SVC",
        "kernel": "poly",
        "regularization": 1.0,
        "gamma": "scale",
        "max_iter": 1000,
        "random_state": 42,
    },
    "SVC_LINEAR": {
        "model_name": "SVC",
        "kernel": "linear",
        "regularization": 1.0,
        "gamma": "scale",
        "max_iter": 1000,
        "random_state": 42,
    },
    "SVC_SIGMOID": {
        "model_name": "SVC",
        "kernel": "sigmoid",
        "regularization": 1.0,
        "gamma": "scale",
        "max_iter": 1000,
        "random_state": 42,
    }
}
