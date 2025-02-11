# Image Classifier with Flask API and Streamlit Frontend

This project demonstrates an **Image Classifier** using a Flask API for prediction and a Streamlit application as a user interface. The classifier uses a **Random Forest model** trained on the **Iris dataset**, and the predictions are made based on basic image features.

---
## **Project Architecture**

### **Architecture Diagram**
```
[Streamlit UI]
     |
     v
[Flask API]
     |
     v
[Random Forest Model (iris_classifier.joblib)]
     |
     v
[Prediction Returned to Streamlit]

```

1. **Model Training**: A `RandomForestClassifier` is trained using the Iris dataset and saved as a `.joblib` file.
2. **Flask API**: Exposes a `/predict` endpoint to classify uploaded images.
3. **Streamlit Frontend**: Provides an interactive UI to upload images and get predictions from the Flask API.
4. **Dockerized Services**: Flask and Streamlit are containerized using Docker for easy deployment.

---
## **Model Training**

The Random Forest model is trained on the **Iris dataset** and saved as a `joblib` file in the `joblib` directory. Here is the code for training and saving the model:


---
## **Project Structure**
```
image-classifier/
│
├── joblib/                      # Directory containing the .joblib file
│   └── iris_classifier.joblib
├── flask_app/                   # Flask application directory
│   ├── app.py                   # Flask API code
│   ├── Dockerfile               # Dockerfile for Flask
│   └── requirements.txt         # Dependencies for Flask
├── streamlit_app/               # Streamlit application directory
│   ├── app.py                   # Streamlit UI code
│   ├── Dockerfile               # Dockerfile for Streamlit
│   └── requirements.txt         # Dependencies for Streamlit
└── docker-compose.yml           # Docker Compose configuration
```

---
## **Docker Setup and Commands**

### **Docker Compose Configuration**
The project uses Docker Compose to manage the Flask and Streamlit services.

**`docker-compose.yml`:**

### **Docker Commands**
- **Build the containers:**
  ```bash
  docker-compose build
  ```
- **Start the services:**
  ```bash
  docker-compose up
  ```
- **Stop the services:**
  ```bash
  docker-compose down
  ```

---
## **Control Flow**
1. **Model Training:** A Random Forest model is trained and saved as a `.joblib` file.
2. **Flask API:** The API loads the `.joblib` file and exposes a `/predict` endpoint.
3. **Streamlit Frontend:** Users upload images, and the app sends the image to the Flask API for prediction.
4. **Dockerized Services:** Both Flask and Streamlit run as separate Docker containers.

---
## **Screenshots**

### **Streamlit Interface**
![Streamlit Interface](Screenshot%202025-02-11%20201839.jpg)

### **Prediction Output**
![Prediction Output](Screenshot%202025-02-11%20201907.jpg)
![Prediction Details](Screenshot%202025-02-11%20201926.jpg)

### **Docker Logs**
![Docker Logs](Screenshot%202025-02-11%20201945.jpg)

---
## **How to Run the Project**

1. **Train the Model:**
   ```bash
   python training_script.py
   ```
   This will generate the `iris_classifier.joblib` file in the `joblib` directory.

2. **Build and Run the Docker Containers:**
   ```bash
   docker-compose build
   docker-compose up
   ```

3. **Access the Applications:**
   - **Flask API:** `http://127.0.0.1:5000`
   - **Streamlit App:** `http://127.0.0.1:8501`

---
## **Future Improvements**
- Train a custom model specifically for image classification.
- Deploy the service on cloud platforms like AWS or GCP.
- Add authentication to the API.

---
## **License**
N/A

