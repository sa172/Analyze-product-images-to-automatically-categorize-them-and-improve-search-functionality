from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# retrieve environment variables
ENDPOINT = os.getenv("VISION_TRAINING_ENDPOINT")
PREDICTION_ENDPOINT = os.getenv("VISION_TRAINING_ENDPOINT")
# ENDPOINT1= os.getenv("VISION_PREDICTION_ENDPOINT")
training_key = os.getenv("VISION_TRAINING_KEY")
prediction_key = os.getenv("VISION_PREDICTION_KEY")
prediction_resource_id = os.getenv("VISION_PREDICTION_RESOURCE_ID")

#Authneticating the client
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

def create_project(project_name):
    return trainer.create_project(project_name)

def add_images_from_directory(project_id, directory, tag):
    image_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                with open(os.path.join(root, file), "rb") as image_contents:
                    image_list.append(ImageFileCreateEntry(name=file, contents=image_contents.read(), tag_ids=[tag.id]))
    
    for i in range(0, len(image_list), 64):
        batch = image_list[i:i+64]
        trainer.create_images_from_files(project_id, ImageFileCreateBatch(images=batch))

def train_project(project_id):
    iteration = trainer.train_project(project_id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project_id, iteration.id)
        st.text(f"Training status: {iteration.status}")
        time.sleep(10)
    return iteration

# section for checking the how many projects are there for free tier account
def get_or_create_project(project_name):
    # List existing projects
    projects = trainer.get_projects()
    
    # Check if the project already exists
    for project in projects:
        if project.name == project_name:
            st.write(f"Using existing project: {project_name}")
            return project

    # If less than 2 projects exist, create a new project
    if len(projects) < 2:
        st.write(f"Creating new project: {project_name}")
        return trainer.create_project(project_name)
    else:
        st.error("Cannot create more than 2 projects with the current subscription. Please delete an existing project or upgrade your subscription.")
        return None

def main():
    st.title("Custom Vision Image Classifier")

    # Training Section
    st.header("Train the Model")
    project_name = st.text_input("Enter a name for the project:")
    category_name = st.text_input("Enter the category name for the images:")
    training_dir = st.text_input("Enter the full path to the directory containing training images:")

    if st.button("Train Model"):
        with st.spinner("Creating project and uploading images..."):
            # project = create_project(project_name)
            project = get_or_create_project(project_name)
            if project is None:
                return 
            category_tag = trainer.create_tag(project.id, category_name)
            add_images_from_directory(project.id, training_dir, category_tag)

        with st.spinner("Training the model..."):
            iteration = train_project(project.id)

        with st.spinner("Publishing the model..."):
            publish_iteration_name = "Iteration1"
            trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
        
        st.success("Model trained and published successfully!")
        st.session_state['project_id'] = project.id
        st.session_state['publish_iteration_name'] = publish_iteration_name

    # Prediction Section
    st.header("Make Predictions")
    if 'project_id' in st.session_state and 'publish_iteration_name' in st.session_state:
        uploaded_files = st.file_uploader("Choose image(s) for prediction", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image_bytes = uploaded_file.read()
                # st.image(image_bytes, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
                
                with st.spinner("Making prediction..."):
                    try:
                        
                        results = predictor.classify_image(st.session_state['project_id'], st.session_state['publish_iteration_name'], image_bytes)
                        st.subheader(f"Predictions for {uploaded_file.name}:")
                        for prediction in results.predictions:
                            st.write(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
    else:
        st.warning("Please train a model first before making predictions.")

if __name__ == "__main__":
    main()

# #creating the new vision project

# publish_iteration_name = "Iteration 1"

# credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
# trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# # # Create a new project
# # print ("Creating project...")
# # project_name = uuid.uuid4()
# # project = trainer.create_project(project_name)

# # Get the image directory path from the user
# image_dir = input("Enter the full path to the directory containing the images: ")
# category_name = input("Enter the category name for these images (e.g., 'Apparel_Boys'): ")

# # Create a new project
# print("Creating project...")
# project = trainer.create_project(f"{category_name} Classification")

# # Create tag for the category
# category_tag = trainer.create_tag(project.id, category_name)

# # Function to add images from the specified directory
# def add_images_from_directory(directory, tag):
#     image_list = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 with open(os.path.join(root, file), "rb") as image_contents:
#                     image_list.append(ImageFileCreateEntry(name=file, contents=image_contents.read(), tag_ids=[tag.id]))
#     return image_list

# # Add images
# print("Adding images...")
# image_list = add_images_from_directory(image_dir, category_tag)

# # Upload images
# print(f"Uploading {len(image_list)} images...")
# batch_size = min(64, len(image_list))  # Adjust batch size based on number of images
# for i in range(0, len(image_list), batch_size):
#     batch = image_list[i:i+batch_size]
#     try:
#         upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=batch))
#         if not upload_result.is_batch_successful:
#             print(f"Batch starting at index {i} had some failures.")
#         for image in upload_result.images:
#             if image.status != "OK":
#                 print(f"Image {image.source_url} failed to upload. Status: {image.status}")
#     except Exception as e:
#         print(f"An error occurred while uploading batch starting at index {i}: {str(e)}")

# print("Image upload process completed.")

# # # Upload images
# # print(f"Uploading {len(image_list)} images...")
# # for i in range(0, len(image_list), 64):
# #     batch = image_list[i:i+64]
# #     upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=batch))
# #     if not upload_result.is_batch_successful:
# #         print("Image batch upload failed.")
# #         for image in upload_result.images:
# #             print("Image status: ", image.status)
# #         exit(-1)

# # Train the project
# print("Training the project...")
# iteration = trainer.train_project(project.id)
# while iteration.status != "Completed":
#     iteration = trainer.get_iteration(project.id, iteration.id)
#     print("Training status: " + iteration.status)
#     time.sleep(10)

# # Publish the iteration to the project endpoint
# # publish_iteration_name = f"{category_name}Model"
# trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
# print("Model trained and published!")

# print(project.id)
# print(publish_iteration_name)
# print(prediction_resource_id)

# # Set up the prediction client
# prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
# if prediction_credentials:
#     print("yes")
# predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
# if predictor:
#     print("yes")


# print(project.id)
# print(publish_iteration_name)

# # Perform a test prediction
# try:
    
#     test_image_path = input("Enter the full path to a test image: ")
#     with open(test_image_path, "rb") as image_contents:
#         results = predictor.classify_image(project.id, publish_iteration_name, image_contents.read())
# except Exception as e:
#     print(f"Error processing {test_image_path}: {str(e)}")
    

# # Display the results
# print("Results of the test prediction:")
# for prediction in results.predictions:
#     print(f"\t{prediction.tag_name}: {prediction.probability * 100:.2f}%")

# print("Done!")
