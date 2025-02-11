import joblib
import os
from PIL import Image
import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset to map prediction class names
iris = load_iris()

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Construct the full path to the 'joblib' directory and model file
joblib_dir = os.path.join(parent_dir, "joblib")
model_path = os.path.join(joblib_dir, "iris_classifier.joblib")

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
else:
    # Load the saved model from the 'joblib' directory in the parent directory
    loaded_model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Read the flower image from the current directory
    current_dir = os.getcwd()
    image_file = os.path.join(current_dir, "flower.jpg")

    if not os.path.exists(image_file):
        print(f"Image file not found at {image_file}")
    else:
        # Open and preprocess the image
        image = Image.open(image_file).convert("RGB")
        image_resized = image.resize((128, 128))
        image_array = np.array(image_resized)

        # Extract features: mean of RGB channels and image size
        r_mean = image_array[:, :, 0].mean()
        g_mean = image_array[:, :, 1].mean()
        b_mean = image_array[:, :, 2].mean()
        size = image_array.size  # Total number of pixels

        # Prepare the feature array
        sample_data = [[r_mean, g_mean, b_mean, size]]

        # Predict using the loaded model
        try:
            prediction = loaded_model.predict(sample_data)

            # Print the predicted class name
            predicted_class = iris.target_names[prediction][0]
            print(f"Predicted class: {predicted_class}")
        except Exception as e:
            print(f"Error during prediction: {e}")
