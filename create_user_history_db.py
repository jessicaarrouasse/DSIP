import os
import pandas as pd
import argparse

def user_history_feature_extraction(train_df):
    train_data = train_df.copy()
    # Convert DateTime to datetime type for proper handling
    train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])
    # Sort train_data by DateTime
    train_data = train_data.sort_values(by='DateTime')
    # Aggregate user-level features
    user_history_db = (
        train_data.groupby('user_id')
        .agg(
            #last_visit=('DateTime', 'max'),  # Last visit time
            total_sessions=('session_id', 'count'),  # Total number of sessions
            last_campaign_id=('campaign_id', lambda x: x.iloc[-1]),  # Last campaign interacted with
            click_rate=('is_click', 'mean'),  # Click-through rate
        )
        .reset_index()
    )
    return user_history_db

def main(train_data_path):
    train_df = pd.read_csv(train_data_path)
    user_history_db = user_history_feature_extraction(train_df)
    user_history_db.to_csv("./data/user_history_db.csv", index=False)
    print("User history database created!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--train-data-path", type=str)
    args = parser.parse_args()
    main(train_data_path)