import streamlit as st
import os
import uuid
import time
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, ImageFileCreateBatch
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv

load_dotenv()
# retrieve environment variables
ENDPOINT = os.environ("VISION_TRAINING_ENDPOINT")
training_key = os.environ("VISION_TRAINING_KEY")
prediction_key = os.environ("VISION_PREDICTION_KEY")
prediction_resource_id = os.environ("VISION_PREDICTION_RESOURCE_ID")

# Set up the credentials
def get_credentials(training_key, prediction_key, endpoint):
    training_credentials = CognitiveServicesCredentials(training_key)
    prediction_credentials = CognitiveServicesCredentials(prediction_key)
    trainer = CustomVisionTrainingClient(endpoint, training_credentials)
    predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)
    return trainer, predictor

# Function to create and train the model
def create_and_train_model(trainer, project_name, base_image_location, category_names):
    print("Creating project...")
    project = trainer.create_project(project_name)
    
    # Create tags for each category
    tags = {}
    for category in category_names:
        tags[category] = trainer.create_tag(project.id, category)

    print("Adding images...")
    image_list = []

    # Add images for each category
    for category in category_names:
        category_path = os.path.join(base_image_location, category)
        for image_num in range(1, 11):  # Adjust range as necessary
            file_name = f"{category.lower().replace(' ', '_')}_{image_num}.jpg"
            file_path = os.path.join(category_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, "rb") as image_contents:
                    image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[tags[category].id]))

    # Upload images
    upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
    if not upload_result.is_batch_successful:
        st.error("Image batch upload failed.")
        for image in upload_result.images:
            st.write(f"Image status: {image.status}")
        return None
    
    # Train the project
    st.text("Training...")
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        st.text(f"Training status: {iteration.status}")
        time.sleep(10)

    # Publish the iteration
    publish_iteration_name = "classifyModel"
    trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
    st.success("Training complete and model published!")

# Function to predict image category
def predict_image(predictor, project_id, publish_iteration_name, uploaded_image):
    results = predictor.classify_image(project_id, publish_iteration_name, uploaded_image.read())
    return results

# Streamlit app layout
st.title("Custom Vision Model Training and Prediction")

# Input fields for project name and category names
project_name = st.text_input("Project Name", value="MyCustomProject")
category_names = st.text_input("Category Names (comma separated)", value="Hemlock, Japanese Cherry")
base_image_location = st.text_input("Base Image Location", value="path/to/your/images")

# Upload images for prediction
uploaded_image = st.file_uploader("Upload an image to predict", type=["jpg", "jpeg", "png"])

# Input fields for Azure credentials
training_key = st.text_input("Training Key", type="password")
prediction_key = st.text_input("Prediction Key", type="password")
endpoint = st.text_input("Endpoint URL")

if st.button("Train Model"):
    if training_key and prediction_key and endpoint and project_name and base_image_location:
        category_names_list = [name.strip() for name in category_names.split(",")]
        trainer, predictor = get_credentials(training_key, prediction_key, endpoint)
        create_and_train_model(trainer, project_name, base_image_location, category_names_list)
    else:
        st.error("Please fill in all fields.")

if st.button("Predict Image"):
    if uploaded_image and training_key and prediction_key and endpoint and project_name:
        category_names_list = [name.strip() for name in category_names.split(",")]
        trainer, predictor = get_credentials(training_key, prediction_key, endpoint)
        results = predict_image(predictor, project_name, "classifyModel", uploaded_image)
        st.write("Prediction Results:")
        for prediction in results.predictions:
            st.write(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
    else:
        st.error("Please upload an image and fill in all fields.")
