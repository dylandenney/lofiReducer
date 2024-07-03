import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load all saved MFCC files for commercials and non-commercials
commercial_files = glob.glob("data/commercial_mfcc_*.npy")
non_commercial_files = glob.glob("data/non_commercial_mfcc_*.npy")

# Load and label the data
X = []
y = []

def load_and_append(file_list, label):
    for file in file_list:
        try:
            mfccs = np.load(file)
            if mfccs.size == 0:
                print(f"Warning: File {file} is empty.")
                continue
            X.append(mfccs.flatten())  # Flatten the MFCC array to a 1D array
            y.append(label)
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")

load_and_append(commercial_files, 1)  # Label for commercials
load_and_append(non_commercial_files, 0)  # Label for non-commercials

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Print some debugging information
print(f"Loaded {len(X)} samples")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Rest of your script remains the same...
