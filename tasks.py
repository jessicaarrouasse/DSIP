from invoke import task

@task
def pipeline(context):
	context.run("python preprocess.py --csv-path train_dataset_full.csv")
	context.run("python train.py decision_tree")
	context.run("python predict.py decision_tree")
