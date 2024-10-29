import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Set the path to the dataset
train_dir = 'path/to/train'  # Update this to the path where your dataset is located
image_size = (64, 64)  # Resize images to this size

# Function to load and preprocess the images
def load_data(train_dir):
    images = []
    labels = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.jpg'):  # Ensure we are only reading image files
            label = 'dog' if 'dog' in filename else 'cat'
            img_path = os.path.join(train_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load the images and labels
images, labels = load_data(train_dir)

# Flatten the images and normalize pixel values
images = images.reshape(len(images), -1)  # Flatten each image
images = images.astype('float32') / 255.0  # Normalize

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')  # You can also experiment with 'rbf' or 'poly'
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Function to visualize predictions
def visualize_predictions(X, y_true, y_pred, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(image_size[0], image_size[1], 3))
        plt.title(f'True: {le.inverse_transform([y_true[i]])[0]}\nPred: {le.inverse_transform([y_pred[i]])[0]}')
        plt.axis('off')
    plt.show()

# Visualize some predictions
visualize_predictions(X_test, y_test, y_pred, n=10)
