import streamlit as st
import requests
from PIL import Image

# Function to send a POST request to the Flask API and get the prediction
def get_prediction_from_api(image_path):
    url = "http://172.18.0.2:5000/predict"
    with open(image_path, "rb") as image_file:
        response = requests.post(url, files={"file": ("flower.jpg", image_file, "image/jpeg")})

    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get("prediction", "Error in response")
    else:
        error_message = response.json().get("error", "Unknown error")
        return f"Error from Flask API: {error_message}"

# Streamlit UI
st.title("Image Classifier with Flask API")
st.write("Upload an image and get a prediction from the Flask API.")
st.write("### Class Labels:")
st.write("""
- **0 → Setosa**  
- **1 → Versicolor**  
- **2 → Virginica**
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())

    # Call the prediction function
    with st.spinner("Getting prediction from Flask API..."):
        prediction = get_prediction_from_api("temp_image.jpg")
        st.success(f"Prediction: {prediction}")
        # Display the prediction result
        st.write(f"### Prediction: {prediction}")
        if prediction == "0":
            st.success("This flower is likely **Setosa**.")
        elif prediction == "1":
            st.success("This flower is likely **Versicolor**.")
        elif prediction == "2":
            st.success("This flower is likely **Virginica**.")
        else:
            st.error("Could not classify the image.")
