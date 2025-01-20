import argparse
import os
import pickle

import pandas as pd
from constants import DECISION_TREE


def get_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_model(model_name):
    # load
    with open(f'models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model_name, csv_path):
    predict_df = get_data(csv_path)
    model = load_model(model_name)
    predictions = model.predict(predict_df)



    print("Predict the model")



if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Trainer")
   parser.add_argument("-m", "--model_name", default=DECISION_TREE)
   parser.add_argument("-e", "--csv_path", default=DECISION_TREE)

   args = parser.parse_args()
   predict(args.model_name, args.csv_path)
