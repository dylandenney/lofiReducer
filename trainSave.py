import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load all saved MFCC files for commercials and non-commercials
commercial_files = glob.glob("data/commercial_mfcc_*.npy")
non_commercial_files = glob.glob("data/non_commercial_mfcc_*.npy")

# Load and label the data
X = []
y = []

for file in commercial_files:
    mfccs = np.load(file)
    X.append(mfccs.flatten())  # Flatten the MFCC array to a 1D array
    y.append(1)  # Label for commercials

for file in non_commercial_files:
    mfccs = np.load(file)
    X.append(mfccs.flatten())  # Flatten the MFCC array to a 1D array
    y.append(0)  # Label for non-commercials

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a classifier (Random Forest in this case)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a detailed classification report
report = classification_report(y_test, y_pred, target_names=['Non-commercial', 'Commercial'])
print(report)

# Save the trained model to a file
joblib.dump(classifier, 'models/trained_model.pkl')

print("Trained model saved to 'models/trained_model.pkl'")

