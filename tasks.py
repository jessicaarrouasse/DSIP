from invoke import task

@task
def pipeline(context):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run("python train.py --models-path ./data/ --model Random_Forest")
	context.run("python predict.py --model-path ./models/Random_Forest.pkl --test-data-path ./data/X_test.csv")
	context.run("python results.py -pp predictions/Random_Forest_predictions.csv -ppp predictions/Random_Forest_proba_predictions.csv -gt data/y_test.csv")
