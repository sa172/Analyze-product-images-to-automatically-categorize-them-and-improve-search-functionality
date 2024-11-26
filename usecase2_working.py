import streamlit as st
import requests
import os
from PIL import Image
import io

# Azure Custom Vision settings
endpoint ="https://sarocustomvson-prediction.cognitiveservices.azure.com/".rstrip("/") # Replace with your Azure endpoint
PREDICTION_KEY ="6ff828e684564a8da583cd99ffd551e8"  # Replace with your prediction key
project_id ="49686b3e-3304-4c62-8b1e-cca8d1e5d052"  # Your project id
published_model_name ="Iteration1"  # Replace with your published model name
published_url = "https://sarocustomvson-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/49686b3e-3304-4c62-8b1e-cca8d1e5d052/classify/iterations/Iteration1/image"

# Function to make predictions
def predict_image(image_data):
    headers = {
        "Content-Type": "application/octet-stream",
        "Prediction-Key": PREDICTION_KEY
    }
    print(headers)
    url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{published_model_name}/image"
    print(url)

    print("both are same")
    response = requests.post(url, headers=headers, data=image_data)
    print(response)


    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error in prediction: {response.status_code}, {response.text}")
        return None

# Streamlit app
st.title("Azure Custom Vision Image Prediction")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    print(image_data)
    image = Image.open(io.BytesIO(image_data))
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predictions = predict_image(image_data)
        print("yes")
        st.write(predictions)
        
        if predictions:
            st.write("Predictions:")
            for prediction in predictions['predictions']:
                st.write(f"{prediction['tagName']} - Probability: {prediction['probability']:.2f}")

# # Function to make predictions
# def predict_image(image_path):
#     headers = {
#         "Content-Type": "application/octet-stream",
#         "Prediction-Key": prediction_key
#     }

#     url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{published_model_name}/image"

#     with open(image_path, "rb") as image_data:
#         response = requests.post(url, headers=headers, data=image_data)

#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Error in prediction: {response.status_code}, {response.text}")
#         return None

# # Main execution
# if __name__ == "__main__":
#     image_path = input("Enter the path to the image file: ")  # Example: "path/to/your/image.jpg"
    
#     if os.path.exists(image_path):
#         predictions = predict_image(image_path)

#         if predictions:
#             print("Predictions:")
#             for prediction in predictions['predictions']:
#                 print(f"{prediction['tag']} - Probability: {prediction['probability']:.2f}")
#     else:
#         print("The provided image path does not exist.")
        
# # If you want to upload images from specific folders
# folder_path = "Images"  # Set your folder path
# if st.button("Upload Images from Folders"):
#     hemlock_images = os.listdir(os.path.join(folder_path, "Hemlock"))
#     japanese_cherry_images = os.listdir(os.path.join(folder_path, "Japanese_Cherry"))
#     st.write("Hemlock Images:", hemlock_images)
#     st.write("Japanese Cherry Images:", japanese_cherry_images)
