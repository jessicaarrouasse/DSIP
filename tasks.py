from invoke import task

@task
def pipeline(context):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run("python train.py -m naive_bayes")
	context.run("python predict.py -m naive_bayes")
	# Calculate metrics
	context.run("python results.py -m naive_bayes")

	'''models = ['naive_bayes', 'xgboost', 'lightgbm']

	for model in models:
		print(f"Running pipeline for {model}...")

		# Train the model
		context.run(f"python train.py -m {model}")

		# Predict using the model
		context.run(f"python predict.py -m {model}")

		# Calculate metrics
		context.run(f"python results.py -m {model}")
	'''