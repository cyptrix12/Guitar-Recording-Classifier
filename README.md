# Guitar Recording Classifier – Feature Extraction & kNN Classification

This project is part of the group work titled **"Guitar Recording Corpus for Automatic Instrument Model Recognition"** developed at Gdańsk University of Technology (WETI PG). It provides a basic workflow for extracting audio features (MFCCs) from guitar recordings and classifying them using a k-Nearest Neighbors (kNN) model.

## Structure

The workflow is split into three Python scripts:

---

### 1. `guitar_recording_organizer.py`

This script provides a full graphical user interface (GUI) for organizing guitar recording files.

#### Features

- Allows the user to select input and output folders via a dialog box.
- Automatically detects guitar model names from file names (e.g., `ArgSG_B_otwarta_022_ID1_4.wav` → `ArgSG`), and groups files into corresponding subfolders.
- Supports two modes: **Copy** (preserves source files) and **Move** (relocates them).
- Displays a progress bar and current file status during the operation.
- Provides a log panel with detailed real-time feedback.
- Prevents user from selecting the same or nested input/output folders, to avoid data loss or infinite loops.

#### Typical Workflow

1. Launch the script.
2. Select the folder containing guitar recordings (**it will search recursively**).
3. Select the destination folder for organized files.
4. Choose between Copy or Move mode.
5. Click ***Start*** to begin the process.

---

### 2. `extract_features_and_save.py`

This script processes a large folder of guitar recordings organized by instrument model, extracts MFCC (Mel-Frequency Cepstral Coefficient) features from each `.wav` file, and saves the resulting feature matrix and labels to disk in NumPy format.

#### Input Folder Structure

```
data/
├── ArgSG/
│   ├── file1.wav
│   ├── ...
├── EpiSG/
│   ├── file1.wav
│   ├── ...
...
```

Each subfolder represents a different guitar model class.

#### Output

- `features.npy`: a NumPy array of shape `(n_samples, 13)` with extracted MFCC features
- `labels.npy`: a NumPy array of corresponding string class labels (e.g., `"Gretsch"`, `"HBLP"`)

#### Parameters

- Sample rate: 48000 Hz
- MFCC: 13 coefficients per file (averaged over time)

---

### 3. `train_and_classify_knn.py`

This script loads the previously saved feature and label arrays, encodes and normalizes them, and trains a k-Nearest Neighbors (kNN) classifier. It splits the dataset into training and test sets and visualizes a confusion matrix of the predictions.

#### Key Features

- Uses `LabelEncoder` to convert string labels into integer classes
- Applies `StandardScaler` to normalize feature vectors
- Uses `train_test_split(..., stratify=labels)` to ensure balanced class distribution
- Displays a confusion matrix using `matplotlib`

---

## Requirements

Install the necessary Python packages:

```bash
pip install numpy librosa scikit-learn matplotlib
```

---

## Notes

- If your dataset is large (e.g., multiple GBs), the `.npy` format is highly recommended due to fast I/O and reduced RAM usage.
- For best results, ensure each class (guitar model) has a balanced number of samples.
- Accuracy might drop when increasing test size due to less training data and more diverse test cases.

---

