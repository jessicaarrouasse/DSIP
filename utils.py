import pickle

import numpy as np
import pandas as pd


def get_data(csv_path, header=True):
    df = pd.read_csv(csv_path, header="infer" if header else None)
    return df

def save_numpy_array(df, path):
   np.savetxt(path, df, delimiter=",")

def save_dataframe(df, path):
    df.to_csv(path, index=False)


def load_pickle(model_path):
    with open(model_path, 'rb') as f:
        file = pickle.load(f)
    return file
