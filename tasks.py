from invoke import task

@task
def pipeline(context, model_name):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run(f"python train.py --models-path ./data/ --model {model_name}")
	context.run(f"python predict.py --model-path ./models/{model_name}.pkl --test-data-path ./data/X_test.csv")
	context.run(f"python results.py -pp predictions/{model_name}_predictions.csv -ppp predictions/{model_name}_proba_predictions.csv -gt data/y_test.csv")
