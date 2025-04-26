import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load data
X = np.load("features.npy")
y = np.load("labels.npy")
filenames = np.load("filenames.npy")

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Normalize features using StandardScaler (można zmienić na MinMaxScaler jeśli potrzeba)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X_scaled, y_encoded, filenames, test_size=0.2, random_state=42, stratify=y_encoded
)

# Build Feed-Forward Neural Network (FFN)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the FFN
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {test_accuracy:.2f}")

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Feed-Forward Neural Network")
plt.tight_layout()
plt.show()

# Find indices of misclassified samples
misclassified_indices = np.where(y_test != y_pred)[0]

# Display details of misclassified samples along with file names
for idx in misclassified_indices:
    true_label = le.inverse_transform([y_test[idx]])[0]
    predicted_label = le.inverse_transform([y_pred[idx]])[0]
    file_name = filenames_test[idx]
    print(f"File {file_name}: Real label = {true_label}, label classified = {predicted_label}")