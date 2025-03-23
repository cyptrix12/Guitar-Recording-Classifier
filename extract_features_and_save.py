import os
import librosa
import numpy as np

DATA_PATH = "data"
SAMPLE_RATE = 48000
N_MFCC = 13

features = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            try:
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                features.append(mfcc_mean)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Save to file
np.save("features.npy", features)
np.save("labels.npy", labels)
print("Saved features and labels.")
