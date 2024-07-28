import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Load embeddings and labels
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')

# Ensure embeddings are reshaped correctly if necessary
if len(embeddings.shape) == 3:
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])

print(f"Loaded embeddings shape: {embeddings.shape}")
print(f"Loaded labels shape: {labels.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Reduce dimensionality
pca = PCA(n_components=128)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a K-Nearest Neighbors (KNN) classifier on reduced data
knn_pca = KNeighborsClassifier(n_neighbors=1)
knn_pca.fit(X_train_pca, y_train)

# Predict on the test set
y_pred_pca = knn_pca.predict(X_test_pca)

# Calculate metrics
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, average='weighted')
recall_pca = recall_score(y_test, y_pred_pca, average='weighted')
f1_pca = f1_score(y_test, y_pred_pca, average='weighted')

print(f"PCA Reduced Accuracy: {accuracy_pca:.2f}")
print(f"PCA Reduced Precision: {precision_pca:.2f}")
print(f"PCA Reduced Recall: {recall_pca:.2f}")
print(f"PCA Reduced F1-score: {f1_pca:.2f}")
# This script performs several tasks related to training and evaluating a face recognition model using embeddings and labels

# The code performs several tasks to train and evaluate a face recognition model using pre-computed face embeddings and labels. Initially, it loads these embeddings and labels from .npy files and ensures they are correctly shaped, printing the shapes for verification. The dataset is then split into training and testing sets, with 80% allocated for training and 20% for testing. A K-Nearest Neighbors (KNN) classifier is trained using the training data, and its performance is evaluated on the test set by calculating accuracy, precision, recall, and F1 score. Subsequently, the code applies Principal Component Analysis (PCA) to reduce the dimensionality of the embeddings, which helps in managing the high-dimensional data. The training and test sets are transformed using the PCA model, and another KNN classifier is trained on this reduced data. The model's performance is evaluated again on the PCA-transformed test set using the same metrics. Despite these efforts, the model yields a low accuracy of around 0.08 (8%), which could be attributed to factors such as model overfitting, data quality issues, information loss during dimensionality reduction, or dataset imbalance. To improve the performance, further steps such as increasing the number of neighbors in KNN, hyperparameter tuning, data augmentation, or experimenting with more advanced models could be undertaken.