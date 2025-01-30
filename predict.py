import argparse
import os
import pickle
from utils import get_data, save_numpy_array, ThresholdClassifier

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_predictions(model_name, predictions):
    os.makedirs("predictions", exist_ok=True)
    save_numpy_array(predictions, f"./predictions/{model_name}_predictions.csv")


def main(model_path, test_data_path):
    predict_df = get_data(test_data_path)
    model = load_model(model_path)
    predictions = model.predict(predict_df)
    predictions_proba = model.predict_proba(predict_df)
    model_name = model_path.split("/")[-1].split(".")[0]
    save_predictions(f"{model_name}_test_1st", predictions)
    save_predictions(f"{model_name}_proba_test_1st", predictions_proba)
    print("Done")


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--model-path", type=str)
   parser.add_argument( "--test-data-path", type=str)
   args = parser.parse_args()
   main(args.model_path, args.test_data_path)