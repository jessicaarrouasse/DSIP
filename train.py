import argparse
import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from constants import DECISION_TREE


def get_data():
    df = pd.read_csv("data/train.csv")
    return df

def save_model(model, model_name):
    # save
    os.makedirs("models", exist_ok=True)

    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)


    # load
    # with open('model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)


def train(model_name, epochs):
    train_df = get_data()
    model = LogisticRegression()
    model.fit(train_df)



    print("Train the model")



if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Trainer")
   parser.add_argument("-m", "--model_name", default=DECISION_TREE)
   args = parser.parse_args()
   train(args.model_name, args.epochs)
