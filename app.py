from io import StringIO

import streamlit as st
import pandas as pd
from predict import predict
from preprocess_v2 import preprocess


def main():
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
                    processed_data = preprocess(input_df, "predict")
                    # Generate predictions
                    model_path = "C:/Users/user/Documents/ydata/DSIP/models/Random_Forest.pkl"
                    _, predictions_proba = predict(model_path, processed_data)

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