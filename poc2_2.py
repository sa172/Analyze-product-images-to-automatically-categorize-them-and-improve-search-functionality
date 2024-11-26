import streamlit as st
import os
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image
import time
from dotenv import load_dotenv

load_dotenv()

training_key = os.getenv("project_TRAINING_KEY")
prediction_key = os.getenv("project_PREDICTION_KEY")
endpoint = os.getenv("project_ENDPOINT")
project_id = os.getenv("project_PROJECT_ID")
publish_iteration_name = os.getenv("project_ITERATION_NAME")
prediction_resource_id = os.getenv("project_PREDICTION_RESOURCE_ID")

# Create authentication credentials
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(endpoint, credentials)
# Authenticate the prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, credentials)

# Streamlit app title
st.title("Product Image Classification")

# Inputs for dataset path, category, and project name
dataset_path = st.text_input("Enter the path to your image folder")
category_name = st.text_input("Enter the category name")
project_name = st.text_input("Enter the project name")

def add_new_category_and_train(project_id, category_name, dataset_path):
    # Create a new tag for the new category
    new_tag = trainer.create_tag(project_id, category_name)

if os.path.isdir(dataset_path):
    st.write("Dataset folder found successfully!")

    if st.button("Create and Train Model"):
        # Create a new project in Azure Custom Vision
        project = trainer.create_project(project_name)
        st.write(f"Project '{project_name}' created successfully")

        # Function to train the model for the specified category
        def train_model(project_id):
            # Create a tag for the specific category
            tag = trainer.create_tag(project_id, category_name)

            # Loop through each image in the specified folder and upload
            for image_file in os.listdir(dataset_path):
                image_path = os.path.join(dataset_path, image_file)
                with open(image_path, "rb") as img:
                    trainer.create_images_from_data(project_id, img.read(), [tag.id])

            # Train the model
            iteration = trainer.train_project(project_id)
            while iteration.status != "Completed":
                time.sleep(1)
                iteration = trainer.get_iteration(project_id, iteration.id)
            trainer.publish_iteration(project_id, iteration.id, iteration.name,prediction_resource_id)
            print(iteration.name,prediction_resource_id)
            st.write(f"Model training completed and published as {publish_iteration_name}")

        # Train the model for the specified category
        train_model(project.id)
        
def predict_image(image_file):
    # Read the uploaded file as binary data
    img_data = image_file.read()
    
    # Send the image to the Custom Vision Prediction API
    results = predictor.classify_image(project_id, publish_iteration_name, img_data)
    
    # Display predictions
    for prediction in results.predictions:
        st.write(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")

# Prediction section
st.subheader("Upload an image to predict category")
image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file)
    # st.image(image, caption="Uploaded Image", use_column_width=True)


    # # Predict function
    # def predict_image(image):
    #     with open(image_file.name, "rb") as img_data:
    #         results = predictor.classify_image(project_id, publish_iteration_name, img_data.read())

    #     # Display predictions
    #     for prediction in results.predictions:
    #         st.write(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")

    if st.button("Predict"):
        predict_image(image_file)
        
# st.subheader("Add a new category and retrain the model")

# category_name = st.text_input("Enter new category name")
# dataset_path = st.text_input("Enter the dataset folder path for the new category")

# if category_name and dataset_path:
#     if st.button("Add and Retrain"):
#         add_new_category_and_train(project_id, category_name, dataset_path)