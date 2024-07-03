import subprocess
import numpy as np
import librosa
import time
import os
import os
import psutil

def kill_existing_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the process command line contains 'python' and 'data.py'
            if 'python' in proc.info['name'].lower() and any('data.py' in cmd.lower() for cmd in proc.info['cmdline']):
                print(f"Killing existing process: {proc.info['pid']}")
                psutil.Process(proc.info['pid']).terminate()
                time.sleep(1)  # Give it some time to terminate
                if psutil.Process(proc.info['pid']).is_running():
                    psutil.Process(proc.info['pid']).kill()  # Force kill if it's still running
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

# Call the function to kill existing process
kill_existing_process()

# Command to capture audio from the combined monitor source
parec_command = ["parec", "-d", "combined.monitor", "--raw"]

# Open the subprocess to capture audio data
parec_process = subprocess.Popen(parec_command, stdout=subprocess.PIPE)

buffer_size = 4096  # Adjust buffer size if necessary

def process_audio(data, sr=44100, n_fft=1024):
    audio_data = np.frombuffer(data, dtype=np.int16)
    if len(audio_data) % 2 == 0:
        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, n_fft=n_fft)
    return mfccs

def save_mfccs(mfccs, filename):
    np.save(filename, mfccs)

try:
    print("Starting audio capture...")
    sample_count = 0
    while True:
        data = parec_process.stdout.read(buffer_size)
        if not data:
            break
        mfccs = process_audio(data)
        filename = f"data/commercial_mfcc_{sample_count}.npy"
        save_mfccs(mfccs, filename)
        sample_count += 1

except KeyboardInterrupt:
    print("Stopping...")
finally:
    parec_process.terminate()

