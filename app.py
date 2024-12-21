import streamlit as st
import requests
import pandas as pd

API_URL = "http://159.89.34.211:80/predict"

selected_features = [
    "radius_mean", "texture_mean", "radius_se", 
    "texture_se", "smoothness_mean", "symmetry_mean", "feature_sum"]

st.title("Breast Cancer Classification")
st.write("Interact with the deployed Random Forest model to classify tumor samples as Benign or Malignant.")

st.header("Input Features")
user_inputs = {}
for feature in selected_features[:-1]:
    value = st.number_input(f"Enter value for {feature}:", min_value=0.0, step=0.01)
    user_inputs[feature] = value


user_inputs["feature_sum"] = sum(user_inputs.values())


if st.button("Classify"):

    input_df = pd.DataFrame([user_inputs])
    with st.spinner("Sending data to the API..."):
        try:
            response = requests.post(API_URL, json=input_df.to_dict(orient="records")[0])
            response_data = response.json()
            st.success(f"Prediction: {response_data['prediction']}")
            st.write(f"Confidence: {response_data['probabilities']}")
        except Exception as e:
            st.error(f"Error connecting to the API: {str(e)}")