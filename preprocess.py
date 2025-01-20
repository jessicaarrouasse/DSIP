import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse(csv_path):
   #Load the dataset
   print(f"Location of the file: {csv_path}")
   df = pd.read_csv(csv_path)
   return df

def clean_data(df):
   data = df.drop_duplicates()
   # TODO Removing outliers (IQR Method???)
   # TODO Imputation (dealing with missing values => user_group_id by gender/age_level)
   # TODO Normalizing (StandardScaler ??)
   return data

def split_data(df):
   # Define the feature columns (X) and target column (y)
   # session_id,DateTime,user_id,product,campaign_id,webpage_id,product_category_1
   X = df.loc[:, df.columns != 'is_click']  # Feature columns
   y = df["is_click"]  # Target column

   # Split the dataset into training, testing and validation sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   # validation = 20%, test = 20% and train = 60%
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

   # Combine X and y back into dataframes for train, test and validation set
   train_data = pd.concat([X_train, y_train], axis=1)  # Combine features and target for training data
   test_data = pd.concat([X_test, y_test], axis=1)  # Combine features and target for testing data
   validation_data = pd.concat([X_val, y_val], axis=1)  # Combine features and target for validation data

   return train_data, test_data, validation_data


def save_dataframe(df, filename):
   # Create the 'data' folder if it doesn't exist
   output_folder = "data"
   os.makedirs(output_folder, exist_ok=True)
   file_path = os.path.join(output_folder, filename)
   df.to_csv(file_path, index=False)

def main(csv_path):
   df = parse(csv_path)
   df = clean_data(df)
   train_data, test_data, validation_data = split_data(df)
   save_dataframe(train_data, "train.csv")
   save_dataframe(test_data, "test.csv")
   save_dataframe(validation_data, "validation.csv")

   print("Train, test and validation datasets saved")


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--csv-path", type=str)
   args = parser.parse_args()
   main(args.csv_path)

