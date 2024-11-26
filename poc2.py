from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Get values from environment variables
ENDPOINT = os.getenv("CUSTOM_VISION_ENDPOINT")
PREDICTION_KEY = os.getenv("CUSTOM_VISION_PREDICTION_KEY")
PROJECT_ID = os.getenv("CUSTOM_VISION_PROJECT_ID")

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

def classify_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    with BytesIO() as output:
        img.save(output, format="PNG")
        contents = output.getvalue()

    results = predictor.classify_image(PROJECT_ID, ENDPOINT, contents)

    for prediction in results.predictions:
        print(f"Category: {prediction.tag_name}, Probability: {prediction.probability:.2f}")

# Example usage
image_url = ""
classify_image(image_url)



# # upload and tag images

# base_image_location = os.path.join (os.path.dirname("D:\datasets\data\Apparel\Boys\Images\images_with_product_ids"), "Images")

# print("Adding images...")

# image_list = []

# for image_num in range(1, 11):
#     file_name = "hemlock_{}.jpg".format(image_num)
#     with open(os.path.join (base_image_location, "Apperal", file_name), "rb") as image_contents:
#         image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[apperal_tag.id]))

# for image_num in range(1, 11):
#     file_name = "japanese_cherry_{}.jpg".format(image_num)
#     with open(os.path.join (base_image_location, "Footware", file_name), "rb") as image_contents:
#         image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[footware_tag.id]))

# upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
# if not upload_result.is_batch_successful:
#     print("Image batch upload failed.")
#     for image in upload_result.images:
#         print("Image status: ", image.status)
#     exit(-1)
    
# # training the project
# print ("Training...")
# iteration = trainer.train_project(project.id)
# while (iteration.status != "Completed"):
#     iteration = trainer.get_iteration(project.id, iteration.id)
#     print ("Training status: " + iteration.status)
#     print ("Waiting 10 seconds...")
#     time.sleep(10)

# # The iteration is now trained. Publish it to the project endpoint
# trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
# print ("Done!")

# # Now there is a trained endpoint that can be used to make a prediction
# prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
# predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

# with open(os.path.join (base_image_location, "Test/test_image.jpg"), "rb") as image_contents:
#     results = predictor.classify_image(
#         project.id, publish_iteration_name, image_contents.read())

#     # Display the results.
#     for prediction in results.predictions:
#         print("\t" + prediction.tag_name +
#               ": {0:.2f}%".format(prediction.probability * 100))