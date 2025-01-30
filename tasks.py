from invoke import task

@task
def pipeline(context):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run("python train.py --models-path ./data/ --model AdaBoost")
	context.run("python predict.py --model-path ./models/logistic_regression.pkl --test-data-path ./data/X_test.csv")
	context.run("python results.py -pp predictions/AdaBoost_predictions.csv -ppp predictions/AdaBoost_proba_predictions.csv -gt data/y_test.csv")

@task
def trainsvm(context):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run("python train.py --models-path ./data/ --model svm")
