from invoke import task
from configurations import CONFIGURATIONS

@task
def pipeline(context):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run("python train.py decision_tree")
	context.run("python predict.py decision_tree")
     

@task
def train(ctx, model_name="LogisticRegression", epochs=20, max_steps=20, batch_size=32, learning_rate=0.001, alpha=None, max_depth=None, max_iter=100, n_estimators=100, kernel="rbf", regularization=1.0, gamma="scale", random_state=42):
    """
    Task to train a model with the specified parameters.

    Args:
        ctx: Context object provided by invoke.
        model_name (str): Name of the model to train.
        epochs (int): Number of epochs.
        max_steps (int): Maximum number of steps.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate.
        alpha (float): Alpha parameter for Ridge/Lasso models.
        max_depth (int): Maximum depth for tree-based models.
        max_iter (int): Maximum number of iterations for iterative models.
        n_estimators (int): Number of estimators (trees) for ensemble models.
        kernel (str): Kernel type for SVC.
        regularization (float): Regularization parameter for SVC.
        gamma (str): Gamma parameter for SVC.
        random_state (int): Random state for reproducibility.
    """
    print("Running training step...")
    
    # Construct the command dynamically based on provided arguments
    command = f"python train.py --model-name {model_name} --epochs {epochs} --max-steps {max_steps} --batch-size {batch_size} --learning-rate {learning_rate} --max-iter {max_iter}"

    # Add optional parameters
    if alpha is not None:
        command += f" --alpha {alpha}"
    if max_depth is not None:
        command += f" --max-depth {max_depth}"
    if n_estimators is not None:
        command += f" --n-estimators {n_estimators}"
    if kernel is not None:
        command += f" --kernel {kernel}"
    if regularization is not None:
        command += f" --C {regularization }"
    if gamma is not None:
        command += f" --gamma {gamma}"
    if random_state is not None:
        command += f" --random-state {random_state}"
    
    # Run the command
    ctx.run(command)

@task(name="train-model")
def train_model(ctx, config_name=None):
    """
    Task to train a model with predefined configurations.

    Args:
        ctx: Context object provided by invoke.
        config_name (str): Name of the configuration to use.
    """
    if config_name not in CONFIGURATIONS:
        raise ValueError(f"Configuration '{config_name}' not found. Available configurations: {list(CONFIGURATIONS.keys())}")
    
    # Load the configuration
    config = CONFIGURATIONS[config_name]
    # print(f"Running training step with configuration: {config_name}")
    # print(f"Parameters: {config}")
    
    # Construct the command dynamically based on the configuration
    command = f"python train.py"
    for key, value in config.items():
        if value is not None:
            param = f"--{key.replace('_', '-')}"  # Convert underscores to dashes
            command += f" {param} {value}"
    
    # Run the command
    ctx.run(command)

@task(name="train-grid-search")
def train_grid_search(ctx, model_name=None):
    command = f"python train.py --model-name {model_name} --grid-search {True}"
    ctx.run(command)
