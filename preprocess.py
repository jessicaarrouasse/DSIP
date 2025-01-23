import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Tuple, List, Any

def parse(csv_path):
   #Load the dataset
   print(f"Location of the file: {csv_path}")
   df = pd.read_csv(csv_path)
   return df

def get_unique_mappings(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
   """
   Extract unique mappings between user_group_id, gender, and age_level from known data.
   """
   df_known = df.dropna(subset=['user_group_id', 'gender', 'age_level'])

   # Create (gender, age_level) -> user_group_id mapping
   gender_age_groups = df_known.groupby(['gender', 'age_level'])['user_group_id'].unique()
   gender_age_to_group = {}
   for (g, a), group_ids in gender_age_groups.items():
      if len(group_ids) == 1:
         gender_age_to_group[(g, a)] = group_ids[0]

   # Create user_group_id -> gender mapping
   group_gender = df_known.groupby('user_group_id')['gender'].unique()
   group_to_gender = {}
   for group_id, genders in group_gender.items():
      if len(genders) == 1:
         group_to_gender[group_id] = genders[0]

   # Create user_group_id -> age_level mapping
   group_age = df_known.groupby('user_group_id')['age_level'].unique()
   group_to_age = {}
   for group_id, ages in group_age.items():
      if len(ages) == 1:
         group_to_age[group_id] = ages[0]

   return gender_age_to_group, group_to_gender, group_to_age


def impute_from_group_id(row: pd.Series, group_to_gender: Dict, group_to_age: Dict) -> pd.Series:
   """
   Impute gender and age_level from user_group_id.
   """
   if pd.notna(row['user_group_id']):
      group_id = row['user_group_id']

      # Impute gender if missing
      if pd.isna(row['gender']) and group_id in group_to_gender:
         row['gender'] = group_to_gender[group_id]

      # Impute age_level if missing
      if pd.isna(row['age_level']) and group_id in group_to_age:
         row['age_level'] = group_to_age[group_id]

   return row


def impute_group_id(row: pd.Series, gender_age_to_group: Dict) -> pd.Series:
   """
   Impute user_group_id from gender and age_level.
   """
   if pd.isna(row['user_group_id']) and pd.notna(row['gender']) and pd.notna(row['age_level']):
      key = (row['gender'], row['age_level'])
      if key in gender_age_to_group:
         row['user_group_id'] = gender_age_to_group[key]

   return row


def impute_user_info(row: pd.Series, user_to_gender: Dict, user_to_age: Dict) -> pd.Series:
   """
   Olga: Impute gender and age_level from user_id.
   """
   if pd.isna(row['gender']) and row['user_id'] in user_to_gender:
      row['gender'] = user_to_gender[row['user_id']]

   if pd.isna(row['age_level']) and row['user_id'] in user_to_age:
      row['age_level'] = user_to_age[row['user_id']]

   return row

def impute_demographics(df: pd.DataFrame, direction: str = 'both') -> pd.DataFrame:
   """
   Perform imputation on the DataFrame in specified direction.
   """
   gender_age_to_group, group_to_gender, group_to_age = get_unique_mappings(df)

   # Olga: Create mappings from user_id to gender and age_level
   user_to_gender = df[['user_id', 'gender']].dropna().set_index('user_id')['gender'].to_dict()
   user_to_age = df[['user_id', 'age_level']].dropna().set_index('user_id')['age_level'].to_dict()

   result = df.copy()

   if direction in ['both', 'from_group']:
      # Impute gender and age_level from user_group_id
      result = result.apply(
         lambda row: impute_from_group_id(row, group_to_gender, group_to_age),
         axis=1
      )

   if direction in ['both', 'to_group']:
      # Impute user_group_id from gender and age_level
      result = result.apply(
         lambda row: impute_group_id(row, gender_age_to_group),
         axis=1
      )
   # Olga: Impute gender and age_level from user_id
   result = result.apply(
      lambda row: impute_user_info(row, user_to_gender, user_to_age),
      axis=1
   )

   return result

def iterative_imputation(df: pd.DataFrame) -> pd.DataFrame:
   """
   Perform iterative imputation until convergence or max iterations reached.
   """
   result = df.copy()
   changes = 1
   while changes > 0:
      # Store previous state to check for changes
      previous_nulls = result.isna().sum().sum()
      # Perform one round of imputation
      result = impute_demographics(result, direction='both')
      # Check how many values were filled
      current_nulls = result.isna().sum().sum()
      changes = previous_nulls - current_nulls
   return result

def clean_data(df):
   clean_df = df.drop_duplicates()
   # TODO Removing outliers (IQR Method???)
   # demographics imputation
   imputed_clean_df = iterative_imputation(clean_df)
   # TODO Normalizing (StandardScaler ??)
   return imputed_clean_df

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

