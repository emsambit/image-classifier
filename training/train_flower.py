import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0 = setosa, 1 = versicolor, 2 = virginica

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Create the 'joblib' directory in the parent directory if it doesn't exist
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
joblib_dir = os.path.join(parent_dir, "joblib")

if not os.path.exists(joblib_dir):
    os.makedirs(joblib_dir)

# Save the trained model to the 'joblib' directory in the parent directory
joblib_file = os.path.join(joblib_dir, "iris_classifier.joblib")
joblib.dump(clf, joblib_file)
print(f"Model saved at: {joblib_file}")
