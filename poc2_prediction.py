import os
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv

load_dotenv()

# retrieve environment variables
ENDPOINT = os.getenv("VISION_TRAINING_ENDPOINT")
prediction_key = os.getenv("VISION_PREDICTION_KEY")
# prediction_resource_id = os.getenv("VISION_PREDICTION_RESOURCE_ID")
project_id = os.getenv('CUSTOM_VISION_PROJECT_ID')
publish_iteration_name = os.getenv('CUSTOM_VISION_PUBLISH_ITERATION_NAME')

# Authenticate the prediction client
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

def predict_image(image_path):
    with open(image_path, "rb") as image_contents:
        results = predictor.classify_image(
            project_id, 
            publish_iteration_name,
            image_contents.read()
        )
    
    # Display the results
    print(f"Results for image: {image_path}")
    for prediction in results.predictions:
        print(f"\t{prediction.tag_name}: {prediction.probability * 100:.2f}%")

# Get the image directory path from the user
image_dir = input("Enter the path to the directory containing the images: ")

# Process all images in the directory
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            predict_image(image_path)
            print()  # Add a blank line for readability between images

print("Prediction process completed.")