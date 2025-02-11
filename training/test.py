import joblib
import os
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

    # Sample data for prediction
    sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input: sepal length, sepal width, petal length, petal width

    # Predict using the loaded model
    prediction = loaded_model.predict(sample_data)

    # Print the predicted class name
    predicted_class = iris.target_names[prediction][0]
    print(f"Predicted class: {predicted_class}")
