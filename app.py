import pickle
from io import StringIO

import streamlit as st
import pandas as pd
from predict import predict
from preprocess import preprocess_predict

MODEL_PATH = "./models/Random_Forest.pkl"
image_url = "https://directiveconsulting.com/wp-content/uploads/2019/06/unnamed-1.jpg"

@st.cache_resource
def load_pickle_cached(model_path):
    with open(model_path, 'rb') as f:
        file = pickle.load(f)
    return file


def main():
    st.title('Team: Data Queens')
    # Display the image 
    st.image(image_url, width=300)
    st.title("Click Probability Predictor")
    st.write("Upload your test data CSV file to get click probability predictions")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            input_df = pd.read_csv(uploaded_file)

            # Display sample of input data
            st.subheader("Input Data Preview")
            st.dataframe(input_df.head())

            if st.button("Generate Predictions"):
                with st.spinner("We are calling our model to work its magic 🚀🤖"):
                    # Preprocess the data
                    train_stats = load_pickle_cached("./data/train_stats.pkl")
                    processed_data = preprocess_predict(input_df, train_stats)

                    # Generate predictions
                    model = load_pickle_cached(MODEL_PATH)
                    _, predictions_proba = predict(model, processed_data)

                    # Ensure predictions are between 0 and 1
                    predictions = predictions_proba[:, 1]
                    print("prediction", str(predictions))
                    
                    # Create output file + Save the output to session state
                    st.session_state.predictions = predictions
                    output = pd.DataFrame(predictions, columns=['click_probability'])
                    st.write(f"Output DataFrame shape: {output.shape}")
                    st.session_state.output = output

                    csv_buffer = StringIO()
                    output.to_csv(csv_buffer, index=False, header=False)
                    st.session_state.csv_string = csv_buffer.getvalue()


            if 'predictions' in st.session_state:
                # Display sample predictions
                st.subheader("Sample Predictions")
                st.dataframe(st.session_state.output.head())

                # Create download button
                st.download_button(
                    label="Download Predictions Probabilities",
                    data=st.session_state.csv_string,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                # Apply Threshold to predictions in session state
                predictions = st.session_state.predictions  # Get predictions from session state

                # Threshold Slider
                st.subheader("Hi! you can decide on your own model threshold")
                threshold = st.slider(
                    "Choose a threshold for classification (e.g., 0.5 for default)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                )

                predictions_binary = [1 if prob >= threshold else 0 for prob in predictions]
                total_clicks = sum(predictions_binary)
                total_no_clicks = len(predictions_binary) - total_clicks


                # Create CSV for binary predictions
                predictions_binary_df = pd.DataFrame({'click_probability': st.session_state.predictions, 'prediction': predictions_binary})
                binary_csv_buffer = StringIO()
                predictions_binary_df.to_csv(binary_csv_buffer, index=False, header=False)
                binary_csv_string = binary_csv_buffer.getvalue()

                # Create download button
                st.download_button(
                    label="Download Binary Predictions after Thresholding",
                    data=binary_csv_string,
                    file_name="binary_predictions.csv",
                    mime="text/csv",
                )

                # Display Totals
                st.subheader("Threshold-Based Results")
                st.write(f"Total Predicted 'Click': {total_clicks}")
                st.write(f"Total Predicted 'No-Click': {total_no_clicks}")

                # Calculate Percentages
                total_predictions = len(predictions_binary)
                percent_clicks = (total_clicks / total_predictions) * 100
                percent_no_clicks = (total_no_clicks / total_predictions) * 100

                # Display Percentages
                st.write(f"Percentage of 'Click': {percent_clicks:.2f}%")
                st.write(f"Percentage of 'No-Click': {percent_no_clicks:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure your CSV file follows the required schema.")


if __name__ == "__main__":
    main()