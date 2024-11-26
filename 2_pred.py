import streamlit as st
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
load_dotenv()
import os

# endpoint = os.getenv("VISION_TRAINING_ENDPOINT")
endpoint = os.getenv("VISION_PREDICTION_ENDPOINT")
prediction_key = os.getenv("VISION_PREDICTION_KEY")
published_model_name = os.getenv("VISION_ITERATION_NAME")
project_id =os.getenv("VISION_PROJECT_ID")


# Function to set up the prediction client
def get_prediction_client(prediction_key, endpoint):
    credentials = CognitiveServicesCredentials(prediction_key)
    predictor = CustomVisionPredictionClient(endpoint, credentials)
    print(predictor)
    return predictor

# Function to predict image category
def predict_image(predictor, project_id, published_model_name, image_data):
    results = predictor.classify_image(project_id, published_model_name, image_data)
    print(results)
    return results

# Streamlit app layout
st.title("Custom Vision Model Prediction")

# Input fields for Azure credentials and model details
# project_id = st.text_input("Project ID", value="your_project_id")  # Replace with your project ID
# publish_iteration_name = st.text_input("Published Iteration Name")
# training_key = st.text_input("Prediction Key", type="password")
# endpoint = st.text_input("Endpoint URL")

# Upload images for prediction
uploaded_image = st.file_uploader("Upload an image to predict", type=["jpg", "jpeg", "png"])

if st.button("Predict Image"):
    # if uploaded_image and training_key and endpoint and project_id and publish_iteration_name:
    if uploaded_image is not None:
        image_data = uploaded_image.read()
        predictor = get_prediction_client(prediction_key, endpoint)
        print("2")

        # Debugging prints
        st.write("Using Project ID:", project_id)
        st.write("Using Published Iteration Name:", published_model_name)
        print("going to print 1")
        
        results = predict_image(predictor, project_id, published_model_name, image_data)
        print(results)
        print("1")
        st.write("Prediction Results:")
        for prediction in results.predictions:
            st.write(f"{prediction.tagName}: {prediction.probability * 100:.2f}%")
        # try:
        #     results = predict_image(predictor, project_id, published_model_name, image_data)
        #     print("1")
        #     st.write("Prediction Results:")
        #     for prediction in results.predictions:
        #         st.write(f"{prediction.tagName}: {prediction.probability * 100:.2f}%")
        # except Exception as e:
        #     st.error(f"Error during prediction: {e}")
    else:
        st.error("Please upload an image.")
