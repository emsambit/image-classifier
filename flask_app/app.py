from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
import os
import io

app = Flask(__name__)

# Construct the path to the joblib directory in the parent directory
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_path = os.path.join( "joblib", "iris_classifier.joblib")

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model
model = joblib.load(model_path)
print(f"Model loaded successfully from {model_path}")

@app.route("/")
def home():
    return "Welcome to the Image Classifier API!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    print(file.content_type)
    try:
        # Ensure the file is read only once
        file_bytes = file.read()

        # Check the size of the file
        if len(file_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_resized = image.resize((128, 128))
        image_array = np.array(image_resized)

        # Extract features
        r_mean = image_array[:, :, 0].mean()
        g_mean = image_array[:, :, 1].mean()
        b_mean = image_array[:, :, 2].mean()
        size = image_array.size

        # Prepare feature array for prediction
        sample_data = [[r_mean, g_mean, b_mean, size]]

        # Predict using the loaded model
        prediction = model.predict(sample_data)
        return jsonify({"prediction": str(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    #app.run(debug=True)

'''
curl -X POST http://127.0.0.1:5000/predict -F "file=@C:\AIML\pythonProject\image-classifier\flowerimage\flower.jpg"
'''