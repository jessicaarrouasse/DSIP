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
    st.markdown("""
    <footer style="position:fixed; bottom:0; left:0; width:100%; text-align:center; padding:10px;">
        <p>© Y-DATA 2025 Data Queens - All Rights Reserved</p>
    </footer>
    """, unsafe_allow_html=True)
        
    st.title("Team: Data Queens")
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
                with st.spinner("Processing data and generating predictions..."):
                    # Preprocess the data
                    train_stats = load_pickle_cached("./data/train_stats.pkl")
                    scaler = load_pickle_cached("./data/scaler.pkl")
                    processed_data = preprocess_predict(input_df, train_stats, scaler)

                    # Generate predictions
                    model = load_pickle_cached(MODEL_PATH)
                    _, predictions_proba = predict(model, processed_data)

                    # Ensure predictions are between 0 and 1
                    predictions = predictions_proba[:, 1]
                    print("prediction", str(predictions))
                    # Create output file
                    output = pd.DataFrame(predictions, columns=['click_probability'])
                    st.write(f"Output DataFrame shape: {output.shape}")

                    # Alternative way to create CSV
                    csv_buffer = StringIO()
                    output.to_csv(csv_buffer, index=False, header=False)
                    csv_string = csv_buffer.getvalue()

                    # Create download button
                    st.download_button(
                        label="Download Predictions",
                        data=csv_string,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    # Display sample predictions
                    st.subheader("Sample Predictions")
                    st.dataframe(output.head())


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure your CSV file follows the required schema.")


if __name__ == "__main__":
    main()