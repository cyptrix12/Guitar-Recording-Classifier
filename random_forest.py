import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import randint
import matplotlib.pyplot as plt

# Load data
X = np.load("features.npy")
y = np.load("labels.npy")
filenames = np.load("filenames.npy")

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check class distribution
print("Class distribution:")
print(np.unique(y, return_counts=True))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X_scaled, y_encoded, filenames, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 201),  # Random values from 50 to 200
    'max_depth': randint(5, 16),       # Random values from 5 to 15
    'min_samples_split': randint(5, 21) # Random values from 5 to 20
}

# Display information about parameter ranges
print("Random parameter values will be sampled from the following ranges:")
print("n_estimators: 50-200")
print("max_depth: 5-15")
print("min_samples_split: 5-20")

# Use RandomizedSearchCV to find the best parameters
random_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of random combinations to test
    cv=5,
    n_jobs=-1  # Use all available CPU cores
)
random_search.fit(X_train, y_train)

# Display the best parameters
print("\nBest parameters (RandomizedSearchCV):", random_search.best_params_)

# Use the best model for prediction
rf = random_search.best_estimator_
y_pred = rf.predict(X_test)

# Display test set accuracy
print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Display classification report (precision, recall, F1-score for each class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest (with RandomizedSearchCV)")
plt.tight_layout()
plt.show()

# Display feature importances
print("\nMFCC Feature Importances:")
importances = rf.feature_importances_
for i, importance in enumerate(importances):
    print(f"MFCC Feature {i+1}: {importance:.4f}")

# Display details of misclassified samples
misclassified_indices = np.where(y_test != y_pred)[0]
print("\nMisclassified Samples:")
for idx in misclassified_indices:
    true_label = le.inverse_transform([y_test[idx]])[0]
    predicted_label = le.inverse_transform([y_pred[idx]])[0]
    file_name = filenames_test[idx]
    print(f"File {file_name}: True label = {true_label}, Predicted label = {predicted_label}")