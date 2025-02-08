import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Tuple, List, Any
import logging
import pickle
from sklearn.preprocessing import StandardScaler

from utils import save_dataframe

# Set up logging configuration
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def parse(csv_path):
    # Load the dataset
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


def clean_data_train_mode(df):
    clean_df = df.drop_duplicates()
    clean_df = clean_df.dropna(subset=['is_click'])
    # Ensure datetime is parsed
    clean_df = clean_df.copy()
    clean_df['DateTime'] = pd.to_datetime(clean_df['DateTime'])
    # demographics imputation
    imputed_clean_df = iterative_imputation(clean_df)
    return imputed_clean_df

def clean_data_predict_mode(df):
    # Ensure datetime is parsed
    clean_df = df.copy()
    clean_df['DateTime'] = pd.to_datetime(clean_df['DateTime'])
    # demographics imputation
    imputed_clean_df = iterative_imputation(clean_df)
    return imputed_clean_df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    The function splits the input cleaned df into training, test and validation sets
    while ensuring representativeness and preventing data leakage. Each user_id/session_id
    is assigned to only one set to avoid overlap.
    Args:
       df (pd.DataFrame): The cleaned DataFrame to be split.
    Returns:
       Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for each set.
    '''
    df = df.copy()
    # Separate rows with null user_id
    null_user_data = df[df['user_id'].isnull()]
    non_null_user_data = df[df['user_id'].notnull()]
    # Group by `user_id` for non-null data for stratified splitting
    user_groups = non_null_user_data.groupby('user_id')
    user_df = pd.DataFrame({
        'user_id': user_groups.size().index,
        'is_click': user_groups['is_click'].max(),  # Use max to indicate if the user clicked at least once
    })
    # Split into train and temp (validation + test) based on user_id
    train_users, test_users = train_test_split(
        user_df['user_id'],  # Use user_id for splitting
        test_size=0.2,  # 40% for validation + test
        stratify=user_df['is_click'],  # Stratify by is_click
        random_state=42
    )

    # Create the final train, validation, and test sets
    train_data = non_null_user_data[non_null_user_data['user_id'].isin(train_users)]
    test_data = non_null_user_data[non_null_user_data['user_id'].isin(test_users)]
    # Randomly split null_user_data proportionally
    null_train, null_test = train_test_split(null_user_data, test_size=0.2, random_state=42)
    # Combine splits
    train_data = pd.concat([train_data, null_train], ignore_index=True)
    test_data = pd.concat([test_data, null_test], ignore_index=True)

    return train_data, test_data


def compute_train_stats(train_data: pd.DataFrame) -> dict:
    logging.info("Starting to compute statistics for training data.")
    stats = {}
    os.makedirs('stats', exist_ok=True)
    # Check for NaN values in the grouping columns
    missing_values = train_data[['campaign_id', 'product', 'product_category_1', 'webpage_id']].isnull().sum()
    logging.info(f"Missing values in grouping columns: {missing_values}")
    # Drop rows with NaN in grouping columns
    train_data = train_data.dropna(subset=['campaign_id', 'product', 'product_category_1', 'webpage_id'])
    stats['campaign_ctr'] = train_data.groupby('campaign_id')['is_click'].agg('mean').fillna(0)
    stats['product_popularity'] = train_data.groupby('product')['is_click'].agg('mean').fillna(0)
    stats['product_category_popularity'] = train_data.groupby('product_category_1')['is_click'].agg('mean').fillna(0)
    stats['webpage_ctr'] = train_data.groupby('webpage_id')['is_click'].agg('mean').fillna(0)
    logging.info("Statistics computation completed.")
    return stats


def feature_extraction(df: pd.DataFrame, train_stats: dict = None):
    '''
    The function extracts features and labels from the input DataFrame.
    Args:
       df (pd.DataFrame): The DataFrame containing the data.
       train_stats (dict): A dictionary containing precomputed statistics for the training set.
       scaler (StandardScaler): An optional scaler object for feature scaling. If None, a new scaler will be created.
    Returns:
       dataframe
    '''
    df = df.copy()
    logging.info("Starting feature extraction.")
    # 1. Time-Based Features
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['day_name'] = df['DateTime'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].apply(
        lambda x: 1 if x >= 5 else 0)  # should we map to -1/1 instead of 0/1? depend on the model?
    # Time of day feature - Define bins and labels
    bins = [-1, 5, 11, 17, 21, 24]
    labels = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Early Night']
    # Bin the hours into time periods
    df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)
    # Holiday feature
    # a holiday feature for July 4th
    df['is_holiday'] = (df['DateTime'].dt.month == 7) & (df['DateTime'].dt.day == 4)
    # Convert boolean to 0/1
    df['is_holiday'] = df['is_holiday'].astype(int)  # should we map to -1/1? depened on model
    # drop day_name since it's similar to day_of_week feature (day_of_week is numeric and keeps the ordinal relationship)
    df = df.drop('day_name', axis=1)
    # Encode time_period categorical feature for linear models
    # Map time periods to integers
    time_period_map = {
        'Late Night': 0,
        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Early Night': 4
    }
    df['time_period'] = df['time_period'].map(time_period_map)
    # 2. user features - demographics features
    df['gender_binary'] = df['gender'].map({'Male': 0, 'Female': 1,
                                            None: -1})  # should we handle null here? if we use XGBoost, LightGBM we can skip handle nulls
    df = df.drop('gender', axis=1)
    # 3. User Behavior Features: Aggregate user-level data to capture engagement trends - lets make sure we dont have lekage here when spliting the data to train/test
    user_session_count = df.groupby('user_id')['session_id'].nunique().rename('session_count')
    df = df.merge(user_session_count, on='user_id', how='left')
    avg_user_depth = df.groupby('user_id')['user_depth'].mean().rename('avg_user_depth')
    df = df.merge(avg_user_depth, on='user_id', how='left')
    # 4. Campaign & Product Features
    if train_stats is not None:
        # Use precomputed stats for test set or validation set
        campaign_ctr = df['campaign_id'].map(train_stats['campaign_ctr'])
        product_popularity = df['product'].map(train_stats['product_popularity'])
        product_category_popularity = df['product_category_1'].map(train_stats['product_category_popularity'])
        webpage_ctr = df['webpage_id'].map(train_stats['webpage_ctr'])
    else:
        # Drop rows with NaN in grouping columns before computing stats
        df = df.dropna(subset=['campaign_id', 'product', 'product_category_1', 'webpage_id'])
        # Compute stats for training set
        campaign_ctr = df.groupby('campaign_id')['is_click'].transform('mean').fillna(0)
        product_popularity = df.groupby('product')['is_click'].transform('mean').fillna(0)
        product_category_popularity = df.groupby('product_category_1')['is_click'].transform('mean').fillna(0)
        webpage_ctr = df.groupby('webpage_id')['is_click'].transform('mean').fillna(0)

    df['campaign_ctr'] = campaign_ctr
    df['product_popularity'] = product_popularity
    df['product_category_popularity'] = product_category_popularity
    df['webpage_ctr'] = webpage_ctr
    # 5. Interaction Features
    df['engagement_city'] = df['user_depth'] * df['city_development_index']
    df['campaign_product_ctr'] = df['campaign_ctr'] * df['product_popularity']
    df['product_category_1_age_level'] = df['product_category_1'] * df['age_level']

    logging.info("Feature extraction completed.")
    return df

def compute_scaling(selected_features, df, mode , scaler=None):
    X = df[selected_features]  # .to_numpy()
    y = None
    if mode == "train":
        # Extract selected features and labels
        y = df['is_click']  # .to_numpy():
    if scaler is None:
        scaler = StandardScaler()  # Create a new scaler if not provided
        X[selected_features] = scaler.fit_transform(X)
    else:
        X[selected_features] = scaler.transform(X)  # Transform for test/validation data

    return X, y, scaler


def save_numpy_array(df, filename):
    # Create the 'data' folder if it doesn't exist
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    np.savetxt(file_path, df, delimiter=",")


def save_pickle(data, filename):
    # Create the 'data' folder if it doesn't exist
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    # Save the data as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {filename}")


def load_pickle(filename):
    # Define the folder where the pickles are stored
    input_folder = "data"
    file_path = os.path.join(input_folder, filename)
    # Load the data from the pickle file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {filename} does not exist in the {input_folder} folder.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {filename}")
    return data


def preprocess(csv_path, mode):
    # 1. Clean Data
    if mode == "train":
        df = parse(csv_path)
        df = clean_data_train_mode(df)
        # 2. Split Data
        train_data, test_data = split_data(df)
        # 3. Feature Extraction
        # 3.1 Compute Statistics on the Training Set
        train_stats = compute_train_stats(train_data)
        # save the train_stats as a pickle file
        save_pickle(train_stats, "train_stats.pkl")
        # 3.2 Select features - TODO: consider to move to consts
        selected_features = ['campaign_product_ctr', 'webpage_id', 'product_category_popularity',
                             'product_popularity', 'var_1', 'is_weekend', 'is_holiday', 'session_count',
                             'product_category_1_age_level', 'user_id', 'hour', 'time_period',
                             'gender_binary', 'campaign_ctr', 'webpage_ctr', 'engagement_city']
        # 3.3 Feature Extraction for Training
        train_data = feature_extraction(train_data)
        train_data = train_data.dropna(subset=selected_features) # to experiment with? - random forest handles nulls
        X_train, y_train, scaler = compute_scaling(selected_features, train_data, mode, None)
        # save scaler as a pickle file
        save_pickle(scaler, "scaler.pkl")
        # 3.4 Feature Extraction for Test (using precomputed statistics)
        test_data = feature_extraction(test_data, train_stats)
        test_data = test_data.dropna(subset=selected_features) # to experiment with? - random forest handles nulls
        X_test, y_test, _ = compute_scaling(selected_features, test_data, mode, scaler)

        # Save the features and labels s csv files
        save_dataframe(X_train, "data/X_train.csv")
        save_dataframe(y_train, "data/y_train.csv")
        save_dataframe(X_test, "data/X_test.csv")
        save_dataframe(y_test, "data/y_test.csv")

    else:
        # 1. Clean Data
        df = csv_path
        df = clean_data_predict_mode(df)
        test_data = df.copy()
        # 2. Feature Extraction
        # 2.1 Load Statistics on the Training Set + Scaler
        train_stats = load_pickle("train_stats.pkl")
        scaler = load_pickle("scaler.pkl")
        # 2.2 Select features
        selected_features = ['campaign_product_ctr', 'webpage_id', 'product_category_popularity',
                             'product_popularity', 'var_1', 'is_weekend', 'is_holiday', 'session_count',
                             'product_category_1_age_level', 'user_id', 'hour', 'time_period',
                             'gender_binary', 'campaign_ctr', 'webpage_ctr', 'engagement_city']
        # 2.3 Feature Extraction for Test (using precomputed statistics)
        test_data = feature_extraction(test_data, train_stats)
        X_test, _, _ = compute_scaling(selected_features, test_data, mode, scaler)

        # Save the features and labels as csv files
        save_dataframe(X_test, "data/X_test_1st_process.csv")
        return X_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()
    preprocess(args.csv_path, args.mode)