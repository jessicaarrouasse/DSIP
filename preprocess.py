import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Tuple, List, Any
import logging
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.under_sampling import TomekLinks


class CTRFeatureExtractor:
   """Feature extractor for CTR-specific features to be added to preprocess.py"""

   def __init__(self):
      self.scaler = StandardScaler()

   def extract_features(self, df):
      """Extract CTR-specific features based on analysis insights"""
      df = df.copy()

      # Time-based features
      if 'DateTime' in df.columns:
         df['hour'] = df['DateTime'].dt.hour
         df['day_of_week'] = df['DateTime'].dt.dayofweek
         df['is_peak_hour'] = df['hour'].isin([1, 2])
         df['is_monday'] = df['day_of_week'] == 0
         df['is_night'] = (df['hour'] >= 0) & (df['hour'] <= 4)

      # User behavior features
      if 'user_depth' in df.columns:
         df['is_immediate_action'] = df['user_depth'] == 1
         df['is_researcher'] = df['user_depth'] == 3

      # Age-based interaction features
      if 'age_level' in df.columns and 'user_depth' in df.columns:
         df['is_young_researcher'] = (df['age_level'] == 0) & (df['user_depth'] == 3)
         df['is_young_immediate'] = (df['age_level'] == 0) & (df['user_depth'] == 1)

      # Product features
      if 'product_category_1' in df.columns:
         df['is_category_3'] = df['product_category_1'] == 3
      if 'product' in df.columns:
         df['is_product_J'] = df['product'] == 'J'

      # Campaign features
      if 'campaign_id' in df.columns:
         df['is_top_campaign'] = df['campaign_id'] == 405490
      if 'webpage_id' in df.columns:
         df['is_top_webpage'] = df['webpage_id'] == 60305

      # User group features
      if 'user_group_id' in df.columns:
         df['is_high_ctr_group'] = df['user_group_id'].isin([0, 7, 8])

      # City development features
      if 'city_development_index' in df.columns:
         df['is_developed_city'] = df['city_development_index'] >= 0.8

      return df

   def get_feature_names(self):
      """Return list of feature names added by this extractor"""
      return [
         'hour', 'day_of_week', 'is_peak_hour', 'is_monday', 'is_night',
         'is_immediate_action', 'is_researcher', 'is_young_researcher',
         'is_young_immediate', 'is_category_3', 'is_product_J',
         'is_top_campaign', 'is_top_webpage', 'is_high_ctr_group',
         'is_developed_city'
      ]

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


def analyze_feature_correlations(df: pd.DataFrame, selected_features: List[str]) -> Tuple[pd.DataFrame, plt.Figure]:
   """
   Analyze correlations between features and is_click target variable.

   Args:
       df (pd.DataFrame): Input DataFrame containing features and is_click
       selected_features (List[str]): List of feature names to analyze

   Returns:
       Tuple[pd.DataFrame, plt.Figure]: Correlation matrix and correlation heatmap
   """
   # Initialize CTRFeatureExtractor and extract features
   ctr_extractor = CTRFeatureExtractor()
   df = ctr_extractor.extract_features(df)

   # Process DateTime features
   df['DateTime'] = pd.to_datetime(df['DateTime'])
   df['hour'] = df['DateTime'].dt.hour
   df['day_of_week'] = df['DateTime'].dt.dayofweek
   df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

   # Time of day feature
   bins = [-1, 5, 11, 17, 21, 24]
   labels = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Early Night']
   df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)

   # Holiday feature
   df['is_holiday'] = ((df['DateTime'].dt.month == 7) & (df['DateTime'].dt.day == 4)).astype(int)

   # Encode categorical variables
   df['gender_binary'] = df['gender'].map({'Male': 0, 'Female': 1, None: -1})
   time_period_map = {
      'Late Night': 0,
      'Morning': 1,
      'Afternoon': 2,
      'Evening': 3,
      'Early Night': 4
   }
   df['time_period'] = df['time_period'].map(time_period_map)

   # Add CTR features
   df['campaign_ctr'] = df.groupby('campaign_id')['is_click'].transform('mean')
   df['product_popularity'] = df.groupby('product')['is_click'].transform('mean')
   df['product_category_popularity'] = df.groupby('product_category_1')['is_click'].transform('mean')
   df['webpage_ctr'] = df.groupby('webpage_id')['is_click'].transform('mean')

   # Add interaction features
   df['engagement_city'] = df['user_depth'] * df['city_development_index']
   df['campaign_product_ctr'] = df['campaign_ctr'] * df['product_popularity']
   df['product_category_1_age_level'] = df['product_category_1'] * df['age_level']

   # Calculate user session statistics
   df['session_count'] = df.groupby('user_id')['session_id'].transform('nunique')
   df['avg_user_depth'] = df.groupby('user_id')['user_depth'].transform('mean')

   # Select features for correlation analysis
   features_to_analyze = selected_features + ['is_click']
   correlation_df = df[features_to_analyze].copy()

   # Calculate correlation matrix
   correlation_matrix = correlation_df.corr()

   # Create correlation heatmap
   plt.figure(figsize=(15, 12))
   sns.heatmap(correlation_matrix,
               annot=True,
               cmap='coolwarm',
               center=0,
               fmt='.2f',
               square=True,
               cbar_kws={"shrink": .5})
   plt.title('Feature Correlations with Is_Click')
   plt.xticks(rotation=45, ha='right')
   plt.yticks(rotation=0)
   plt.tight_layout()

   return correlation_matrix, plt.gcf()

def clean_data(df):
   clean_df = df.drop_duplicates()
   clean_df = df.dropna(subset=['is_click'])
   # Ensure datetime is parsed
   clean_df = clean_df.copy()
   clean_df['DateTime'] = pd.to_datetime(clean_df['DateTime'])
   # TODO Removing outliers (IQR Method???)
   # demographics imputation
   imputed_clean_df = iterative_imputation(clean_df)
   # TODO Normalizing (StandardScaler ??)
   return imputed_clean_df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   '''
   The function splits the input cleaned df into training, test and validation sets 
   while ensuring representativeness and preventing data leakage. Each user_id/session_id 
   is assigned to only one set to avoid overlap.
   Args:
      df (pd.DataFrame): The cleaned DataFrame to be split.
   Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for each set.
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
   train_users, temp_users = train_test_split(
      user_df['user_id'],  # Use user_id for splitting
      test_size=0.4,  # 40% for validation + test
      stratify=user_df['is_click'],  # Stratify by is_click
      random_state=42
   )
   # Split the remaining data (temp_users) into validation and test sets
   val_users, test_users = train_test_split(
      temp_users,
      test_size=0.5,  # 50% of 40% -> 20% for validation, 20% for test
      stratify=user_df.loc[temp_users, 'is_click'],  # Stratify by is_click for balance
      random_state=42
   )
   # Create the final train, validation, and test sets
   train_data = non_null_user_data[non_null_user_data['user_id'].isin(train_users)]
   val_data = non_null_user_data[non_null_user_data['user_id'].isin(val_users)]
   test_data = non_null_user_data[non_null_user_data['user_id'].isin(test_users)]
   # Randomly split null_user_data proportionally
   null_train, null_temp = train_test_split(null_user_data, test_size=0.4, random_state=42)
   null_validation, null_test = train_test_split(null_temp, test_size=0.5, random_state=42)
   # Combine splits
   train_data = pd.concat([train_data, null_train], ignore_index=True)
   val_data = pd.concat([val_data, null_validation], ignore_index=True)
   test_data = pd.concat([test_data, null_test], ignore_index=True)
   
   return train_data, test_data, val_data

def compute_train_stats(train_data: pd.DataFrame) -> dict:
   logging.info("Starting to compute statistics for training data.")
   stats = {}
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


def feature_extraction(df: pd.DataFrame, selected_features: List[str], train_stats: dict = None,
                       scaler: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
   '''
   The function extracts features and labels from the input DataFrame.
   Args:
       df (pd.DataFrame): The DataFrame containing the data.
       selected_features (List[str]): The list of selected features to extract.
       train_stats (dict): A dictionary containing precomputed statistics for the training set.
       scaler (StandardScaler): An optional scaler object for feature scaling. If None, a new scaler will be created.
   Returns:
       Tuple[np.ndarray, np.ndarray, StandardScaler]: A tuple containing the features (X), labels (y), and the scaler object.
   '''
   df = df.copy()
   logging.info("Starting feature extraction.")

   # Initialize CTRFeatureExtractor
   ctr_extractor = CTRFeatureExtractor()

   # 1. Add CTR-specific features
   df = ctr_extractor.extract_features(df)

   # 2. Time-Based Features
   df['DateTime'] = pd.to_datetime(df['DateTime'])
   df['hour'] = df['DateTime'].dt.hour
   df['day_of_week'] = df['DateTime'].dt.dayofweek
   df['day_name'] = df['DateTime'].dt.day_name()
   df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

   # Time of day feature
   bins = [-1, 5, 11, 17, 21, 24]
   labels = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Early Night']
   df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)

   # Holiday feature
   df['is_holiday'] = (df['DateTime'].dt.month == 7) & (df['DateTime'].dt.day == 4)
   df['is_holiday'] = df['is_holiday'].astype(int)

   # Drop day_name and encode time_period
   df = df.drop('day_name', axis=1)
   time_period_map = {
      'Late Night': 0,
      'Morning': 1,
      'Afternoon': 2,
      'Evening': 3,
      'Early Night': 4
   }
   df['time_period'] = df['time_period'].map(time_period_map)

   # 3. User features
   df['gender_binary'] = df['gender'].map({'Male': 0, 'Female': 1, None: -1})
   df = df.drop('gender', axis=1)

   # 4. User Behavior Features
   user_session_count = df.groupby('user_id')['session_id'].nunique().rename('session_count')
   df = df.merge(user_session_count, on='user_id', how='left')
   avg_user_depth = df.groupby('user_id')['user_depth'].mean().rename('avg_user_depth')
   df = df.merge(avg_user_depth, on='user_id', how='left')

   # 5. Campaign & Product Features
   if train_stats is not None:
      campaign_ctr = df['campaign_id'].map(train_stats['campaign_ctr'])
      product_popularity = df['product'].map(train_stats['product_popularity'])
      product_category_popularity = df['product_category_1'].map(train_stats['product_category_popularity'])
      webpage_ctr = df['webpage_id'].map(train_stats['webpage_ctr'])
   else:
      df = df.dropna(subset=['campaign_id', 'product', 'product_category_1', 'webpage_id'])
      campaign_ctr = df.groupby('campaign_id')['is_click'].transform('mean').fillna(0)
      product_popularity = df.groupby('product')['is_click'].transform('mean').fillna(0)
      product_category_popularity = df.groupby('product_category_1')['is_click'].transform('mean').fillna(0)
      webpage_ctr = df.groupby('webpage_id')['is_click'].transform('mean').fillna(0)

   df['campaign_ctr'] = campaign_ctr
   df['product_popularity'] = product_popularity
   df['product_category_popularity'] = product_category_popularity
   df['webpage_ctr'] = webpage_ctr

   # 6. Interaction Features
   df['engagement_city'] = df['user_depth'] * df['city_development_index']
   df['campaign_product_ctr'] = df['campaign_ctr'] * df['product_popularity']
   df['product_category_1_age_level'] = df['product_category_1'] * df['age_level']

   logging.info("Feature extraction completed.")

   # Update selected_features to include CTR-specific features
   all_features = selected_features + [f for f in ctr_extractor.get_feature_names() if f not in selected_features]

   # Extract features and labels
   X = df[all_features].to_numpy()
   y = df['is_click'].to_numpy()

   # Feature scaling
   if scaler is None:
      scaler = StandardScaler()
      X = scaler.fit_transform(X)
   else:
      X = scaler.transform(X)

   return X, y, scaler

def save_dataframe(df, filename):
   # Create the 'data' folder if it doesn't exist
   output_folder = "data"
   os.makedirs(output_folder, exist_ok=True)
   file_path = os.path.join(output_folder, filename)
   df.to_csv(file_path, index=False)

def save_pickle(data, filename):
    # Create the 'data' folder if it doesn't exist
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    # Save the data as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {filename}")


def main(csv_path):
   # 1. Clean Data
   df = parse(csv_path)
   df = clean_data(df)

   # 2. Split Data
   train_data, test_data, validation_data = split_data(df)

   # Save the datasets
   save_dataframe(train_data, "train.csv")
   save_dataframe(test_data, "test.csv")
   save_dataframe(validation_data, "validation.csv")
   print("Train, test and validation datasets saved")

   # 3. Feature Extraction
   # 3.1 Compute Statistics on the Training Set
   train_stats = compute_train_stats(train_data)

   # 3.2 Define comprehensive feature list
   selected_features = [
      # Original features
      'campaign_product_ctr',
      'webpage_ctr',
      'product_category_popularity',
      'product_popularity',
      'var_1',
      'is_weekend',
      'is_holiday',
      'session_count',
      'product_category_1_age_level',

      # Time-based features
      'hour',
      'day_of_week',
      'time_period',

      # User features
      'gender_binary', # then I removed it
      'age_level',
      'user_depth',
      'avg_user_depth',

      # Location features
      'city_development_index',

      # Engagement features
      'engagement_city',

      # CTR-specific features from CTRFeatureExtractor
      'is_peak_hour',
      'is_night',
      'is_immediate_action',
      'is_researcher',
      'is_young_researcher',
      'is_category_3',
      'is_top_campaign',
      'is_top_webpage',
      'is_high_ctr_group',
      'is_developed_city'
   ]

   # Add correlation analysis here
   corr_matrix, heatmap = analyze_feature_correlations(df, selected_features)

   # Save correlation outputs
   save_pickle(corr_matrix, "correlation_matrix.pkl")
   heatmap.savefig('data/correlation_heatmap.png')
   plt.close()

   # 3.3 Feature Extraction for Training
   X_train, y_train, scaler = feature_extraction(train_data, selected_features)

   # 3.4 Feature Extraction for Test/Validation (using precomputed statistics)
   X_test, y_test, _ = feature_extraction(test_data, selected_features, train_stats)
   X_validation, y_validation, _ = feature_extraction(validation_data, selected_features, train_stats, scaler=scaler)

   # Drop missing data
   train_mask = ~np.isnan(X_train).any(axis=1)
   X_train = X_train[train_mask]
   y_train = y_train[train_mask]

   val_mask = ~np.isnan(X_validation).any(axis=1)
   X_validation = X_validation[val_mask]
   y_validation = y_validation[val_mask]

   test_mask = ~np.isnan(X_test).any(axis=1)
   X_test = X_test[test_mask]
   y_test = y_test[test_mask]

   # Get gender index and split datasets
   gender_idx = selected_features.index('gender_binary')

   # Split datasets by gender
   # Training set
   male_mask_train = X_train[:, gender_idx] == 0
   female_mask_train = X_train[:, gender_idx] == 1

   X_train_male = np.delete(X_train[male_mask_train], gender_idx, axis=1)
   y_train_male = y_train[male_mask_train]
   X_train_female = np.delete(X_train[female_mask_train], gender_idx, axis=1)
   y_train_female = y_train[female_mask_train]

   # Validation set
   male_mask_val = X_validation[:, gender_idx] == 0
   female_mask_val = X_validation[:, gender_idx] == 1

   X_val_male = np.delete(X_validation[male_mask_val], gender_idx, axis=1)
   y_val_male = y_validation[male_mask_val]
   X_val_female = np.delete(X_validation[female_mask_val], gender_idx, axis=1)
   y_val_female = y_validation[female_mask_val]

   # Test set
   male_mask_test = X_test[:, gender_idx] == 0
   female_mask_test = X_test[:, gender_idx] == 1

   X_test_male = np.delete(X_test[male_mask_test], gender_idx, axis=1)
   y_test_male = y_test[male_mask_test]
   X_test_female = np.delete(X_test[female_mask_test], gender_idx, axis=1)
   y_test_female = y_test[female_mask_test]

   # Apply balancing separately for male and female datasets
   X_train_male_balanced, y_train_male_balanced = balance_dataset(
      X_train_male,
      y_train_male,
      majority_ratio=2.0,
      random_state=42
   )

   X_train_female_balanced, y_train_female_balanced = balance_dataset(
      X_train_female,
      y_train_female,
      majority_ratio=2.0,
      random_state=42
   )

   # Save gender-specific datasets
   save_pickle((X_train_male_balanced, y_train_male_balanced), "X_train_y_train_male.pkl")
   save_pickle((X_train_female_balanced, y_train_female_balanced), "X_train_y_train_female.pkl")
   save_pickle((X_test_male, y_test_male), "X_test_y_test_male.pkl")
   save_pickle((X_test_female, y_test_female), "X_test_y_test_female.pkl")
   save_pickle((X_val_male, y_val_male), "X_validation_y_validation_male.pkl")
   save_pickle((X_val_female, y_val_female), "X_validation_y_validation_female.pkl")

   # Also save the combined balanced datasets for comparison
   save_pickle((np.vstack((X_train_male_balanced, X_train_female_balanced)),
                np.hstack((y_train_male_balanced, y_train_female_balanced))),
               "X_train_y_train_combined_balanced.pkl")

   # Save the feature names (without gender_binary) for reference
   selected_features.remove('gender_binary')
   save_pickle(selected_features, "feature_names.pkl")


def balance_dataset(X: np.ndarray, y: np.ndarray,
                    majority_ratio: float = 5.0,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
   """
   Two-step balancing process:
   1. Cut down majority class to a specified ratio
   2. Apply Tomek Links to clean decision boundary

   Args:
       X (np.ndarray): Feature matrix
       y (np.ndarray): Target labels
       majority_ratio (float): Desired ratio of majority to minority class after cutting
       random_state (int): Random state for reproducibility

   Returns:
       Tuple[np.ndarray, np.ndarray]: Balanced features and labels
   """
   logging.info("Starting two-step balancing process...")

   # Get initial class distribution
   unique_classes, class_counts = np.unique(y, return_counts=True)
   initial_dist = dict(zip(unique_classes, class_counts))
   logging.info(f"Initial class distribution: {initial_dist}")

   # Step 1: Cut majority class
   majority_class = max(initial_dist, key=initial_dist.get)
   minority_class = min(initial_dist, key=initial_dist.get)
   minority_count = initial_dist[minority_class]
   desired_majority_count = int(minority_count * majority_ratio)

   majority_indices = np.where(y == majority_class)[0]
   minority_indices = np.where(y == minority_class)[0]

   np.random.seed(random_state)
   selected_majority_indices = np.random.choice(
      majority_indices,
      size=desired_majority_count,
      replace=False
   )

   selected_indices = np.concatenate([selected_majority_indices, minority_indices])
   X_cut = X[selected_indices]
   y_cut = y[selected_indices]

   cut_dist = dict(zip(*np.unique(y_cut, return_counts=True)))
   logging.info(f"Class distribution after cutting: {cut_dist}")

   # Step 2: Apply Tomek Links
   try:
      tomek = TomekLinks(sampling_strategy='majority')
      X_cleaned, y_cleaned = tomek.fit_resample(X_cut, y_cut)
      final_dist = dict(zip(*np.unique(y_cleaned, return_counts=True)))
      logging.info(f"Final class distribution after Tomek Links: {final_dist}")
      return X_cleaned, y_cleaned
   except ValueError as e:
      logging.error(f"Error applying Tomek Links: {str(e)}")
      logging.warning("Returning cut data without Tomek Links")
      return X_cut, y_cut


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--csv-path", type=str)
   args = parser.parse_args()
   main(args.csv_path)