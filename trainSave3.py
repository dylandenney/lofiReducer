import numpy as np
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter

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

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print dataset shapes for verification
print(f"Original dataset shape: {Counter(y_train)}")
print(f"Resampled dataset shape: {Counter(y_train_resampled)}")

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit Grid Search to the data
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters from Grid Search
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(f"Accuracy: {best_rf.score(X_test, y_test):.2f}")
print(classification_report(y_test, y_pred, target_names=['Non-commercial', 'Commercial']))

# Save the best model
joblib.dump(best_rf, 'models/trained_model-3.pkl')

print("Trained model saved to 'models/trained_model-3.pkl'")

