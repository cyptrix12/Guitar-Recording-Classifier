import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC  #
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X_scaled, y_encoded, filenames, test_size=0.2, random_state=42, 
)

# Train SVM classifier 
svm = SVC(kernel='rbf', C=100, class_weight='balanced', random_state=42, gamma='scale')  
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Display results
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - SVM") 
plt.tight_layout()
plt.show()
misclassified_indices = np.where(y_test != y_pred)[0]

# Display details of misclassified samples along with file names
for idx in misclassified_indices:
    true_label = le.inverse_transform([y_test[idx]])[0]
    predicted_label = le.inverse_transform([y_pred[idx]])[0]
    file_name = filenames_test[idx]
    print(f"File {file_name}: Real label = {true_label}, label classified = {predicted_label}")
    

