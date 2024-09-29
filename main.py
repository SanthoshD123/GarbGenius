from sklearn import svm
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
import joblib  # To save and load models

# Define the Vision Transformer model for feature extraction
class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.vit = torch.hub.load('facebookresearch/dino', 'dino_vitb16', pretrained=True)
        self.vit.eval()

    def forward(self, x):
        return self.vit(x)

# Function to load and preprocess an image
def load_image(img, resized_fac=10):
    img_path = os.path.join(DATASET_PATH, "images", img)
    img = cv2.imread(img_path)
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
    return resized

# Extract features from the dataset using the Vision Transformer model
def get_embedding(model, img_name):
    img = Image.open(os.path.join(DATASET_PATH, "images", img_name)).convert("RGB")
    transform = Compose([Resize((224, 224)), ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(img_tensor)
    return feature.squeeze().numpy()

# Load the dataset
DATASET_PATH = "C:\\Users\\User\\PycharmProjects\\GarbGenius\\archive"
print(os.listdir(DATASET_PATH))

# Corrected file path
df = pd.read_csv(os.path.join(DATASET_PATH, "styles.csv"), nrows=5000)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
df.head(10)

# Extract features for all images in the dataset
vit_model = VisionTransformer()
df_embs = np.array([get_embedding(vit_model, img) for img in df['image']])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_embs, df['articleType'], test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensionality
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Tune parameters
parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train_pca, y_train)

# Train the classifier with the best found parameters
svm_classifier = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], kernel=clf.best_params_['kernel'])
svm_classifier.fit(X_train_pca, y_train)

# Save the trained model
joblib.dump(svm_classifier, 'svm_classifier_model.pkl')

# Predict on the test set
y_pred = svm_classifier.predict(X_test_pca)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy*100}")