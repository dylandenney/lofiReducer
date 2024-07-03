import subprocess
import numpy as np
import time
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
classifier = joblib.load('models/trained_model.pkl')

# Command to capture audio from the combined monitor source
parec_command = ["parec", "-d", "combined.monitor", "--raw"]

# Open the subprocess to capture audio data
parec_process = subprocess.Popen(parec_command, stdout=subprocess.PIPE)

# Threshold and duration settings
consistency_threshold = 40  # Number of consistent frames to consider as a state change
tolerance_threshold = 3  # Tolerance for misclassifications
buffer_size = 4096

# Volume levels
NORMAL_VOLUME = 100
REDUCED_VOLUME = 45  # Adjust this value as needed

# Function to process audio data and extract MFCCs
def process_audio(data, sr=44100, n_fft=1024):
    audio_data = np.frombuffer(data, dtype=np.int16)
    if len(audio_data) % 2 == 0:
        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, n_fft=n_fft)
    return mfccs

# Function to classify MFCC data
def classify_mfcc(mfccs):
    mfccs_flattened = mfccs.flatten().reshape(1, -1)
    prediction = classifier.predict(mfccs_flattened)
    return 'Commercial' if prediction[0] == 1 else 'Non-commercial'

# Function to adjust volume
def adjust_volume(volume_level):
    logging.info(f"Adjusting volume to {volume_level}%")
    subprocess.call(["pactl", "set-sink-volume", "combined", f"{volume_level}%"])
    # Check if the volume was actually changed
    result = subprocess.run(["pactl", "get-sink-volume", "combined"], capture_output=True, text=True)
    logging.info(f"Current volume: {result.stdout.strip()}")

consistent_commercial_frames = 0
consistent_non_commercial_frames = 0
tolerant_commercial_frames = 0
tolerant_non_commercial_frames = 0
state = 'Non-commercial'
last_volume_change_time = time.time()
min_time_between_changes = 5  # Minimum time (in seconds) between volume changes

try:
    logging.info("Starting audio capture...")
    while True:
        data = parec_process.stdout.read(buffer_size)
        if not data:
            break

        mfccs = process_audio(data)
        classification = classify_mfcc(mfccs)

        logging.debug(f"Classification: {classification}, State: {state}")
        logging.debug(f"Commercial frames: {consistent_commercial_frames}, Non-commercial frames: {consistent_non_commercial_frames}")

        current_time = time.time()

        if classification == 'Commercial':
            consistent_commercial_frames += 1
            consistent_non_commercial_frames = 0
            tolerant_commercial_frames += 1
            tolerant_non_commercial_frames = 0
            if consistent_commercial_frames >= consistency_threshold and state != 'Commercial':
                if current_time - last_volume_change_time >= min_time_between_changes:
                    logging.info("Commercial detected, reducing volume...")
                    adjust_volume(REDUCED_VOLUME)  # Reduce volume instead of muting
                    state = 'Commercial'
                    last_volume_change_time = current_time
        else:
            consistent_non_commercial_frames += 1
            consistent_commercial_frames = 0
            tolerant_non_commercial_frames += 1
            tolerant_commercial_frames = 0
            if consistent_non_commercial_frames >= consistency_threshold and state != 'Non-commercial':
                if current_time - last_volume_change_time >= min_time_between_changes:
                    logging.info("Non-commercial detected, restoring volume...")
                    adjust_volume(NORMAL_VOLUME)  # Restore volume to 100%
                    state = 'Non-commercial'
                    last_volume_change_time = current_time

        if tolerant_commercial_frames > tolerance_threshold and state == 'Commercial':
            logging.debug(f"Tolerant commercial frames exceeded threshold: {tolerant_commercial_frames}")
            tolerant_commercial_frames = 0

        if tolerant_non_commercial_frames > tolerance_threshold and state == 'Non-commercial':
            logging.debug(f"Tolerant non-commercial frames exceeded threshold: {tolerant_non_commercial_frames}")
            tolerant_non_commercial_frames = 0

except KeyboardInterrupt:
    logging.info("Stopping...")
finally:
    logging.info("Ensuring volume is restored...")
    adjust_volume(NORMAL_VOLUME)  # Ensure volume is restored when the script is stopped
    parec_process.terminate()

