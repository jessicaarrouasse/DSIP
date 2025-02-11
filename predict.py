import argparse
import os
import pickle
from threshold_classifier import ThresholdClassifier
from utils import get_data, save_numpy_array
import pandas as pd


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_predictions(model_name, predictions):
    os.makedirs("predictions", exist_ok=True)
    save_numpy_array(predictions, f"./predictions/{model_name}_predictions.csv")


def predict(model, predict_df):
    predictions = model.predict(predict_df)
    predictions_proba = model.predict_proba(predict_df)
    return predictions, predictions_proba


# Add user history features to the dataframe
def add_user_history_features(predict_df, user_history_df):
    # Merge the user history data with the prediction data (using user_id as the key)
    merged_df = pd.merge(predict_df, user_history_df, on="user_id", how="left")
    # For users not in the history DB, you can fill NaN values with default values or leave them as is
    return merged_df


def main(model_path, test_data_path):
    predict_df = get_data(test_data_path)
    model = load_model(model_path)
    predictions, predictions_proba = predict(model, predict_df)
    model_name = model_path.split("/")[-1].split(".")[0]
    save_predictions(f"{model_name}", predictions)
    save_predictions(f"{model_name}_proba", predictions_proba)
    print("Done")


def main(model_path, test_data_path, user_history_path):
    # Load test data and user history data
    predict_df = get_data(test_data_path)
    user_history_df = get_data(user_history_path)
    
    # Load models for new users and revisiting users
    model_new_users = load_model(model_path)  # model for new users
    model_revisiting_users = load_model(model_path.replace("new_users", "revisiting_users"))  # model for revisiting users

    # Merge user history data with test data
    # predict_df = add_user_history_features(predict_df, user_history_df)

    # Split the data into revisiting users and new users
    revisiting_users = predict_df[predict_df["user_id"].isin(user_history_df["user_id"])]
    new_users = predict_df[~predict_df["user_id"].isin(user_history_df["user_id"])]

    # Select only the relevant features for prediction
    selected_features = ['campaign_product_ctr', 'webpage_id', 'product_category_popularity',
                         'product_popularity', 'var_1', 'is_weekend', 'is_holiday', 'session_count',
                         'product_category_1_age_level', 'user_id', 'hour', 'time_period',
                         'gender_binary', 'campaign_ctr', 'webpage_ctr', 'engagement_city']
    
    selected_features2 = ['campaign_product_ctr', 'webpage_id', 'product_category_popularity',
                        'product_popularity', 'var_1', 'is_weekend', 'is_holiday', 'session_count',
                        'product_category_1_age_level', 'user_id', 'hour', 'time_period',
                        'gender_binary', 'campaign_ctr', 'webpage_ctr', 'engagement_city',
                        'total_sessions', 'last_campaign_id', 'click_rate']
    
    new_users = new_users[selected_features]
    revisiting_users = revisiting_users[selected_features2]
    # Predict for revisiting users
    predictions_revisiting, proba_revisiting = predict(model_revisiting_users, revisiting_users)

    # Predict for new users
    predictions_new, proba_new = predict(model_new_users, new_users)

    # Add predictions and probabilities to their respective DataFrames
    revisiting_users["predictions"] = predictions_revisiting
    revisiting_users["predictions_proba"] = [prob[1] for prob in proba_revisiting]  # Assuming binary classification
    new_users["predictions"] = predictions_new
    new_users["predictions_proba"] = [prob[1] for prob in proba_new]  # Assuming binary classification

    # Combine predictions into a single DataFrame
    final_predictions_df = pd.concat([revisiting_users, new_users])

    # Ensure the final predictions are in the same order as the original predict_df
    final_predictions_df = final_predictions_df.sort_index()

    # Save only the predictions in the same order as `predict_df`
    final_predictions = final_predictions_df["predictions"]
    final_predictions_proba = final_predictions_df["predictions_proba"]

    # Save the predictions
    model_name = model_path.split("/")[-1].split(".")[0]
    save_predictions(f"{model_name}_final", final_predictions.values)
    save_predictions(f"{model_name}_proba_final", final_predictions_proba.values)
    print("Predictions saved!")


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--model-path", type=str)
   parser.add_argument( "--test-data-path", type=str)
   parser.add_argument("--user-history-path", type=str)
   args = parser.parse_args()
   main(args.model_path, args.test_data_path, args.user_history_path)