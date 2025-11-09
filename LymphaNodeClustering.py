import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# --- 1. Setup Feature Extractor (ResNet50) ---

# Load ResNet50 pre-trained on ImageNet, without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, 
                    input_shape=(224, 224, 3), pooling='avg')

# We use the base_model directly for feature extraction
feature_extractor = base_model
# (You can also use 'base_model.predict(x)' directly)

def load_and_preprocess_image(img_path):
    """Loads, resizes, and preprocesses an image for ResNet50."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(image_paths):
    """Extracts a feature vector for each image in the list."""
    all_features = []
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        preprocessed_img = load_and_preprocess_image(img_path)
        features = feature_extractor.predict(preprocessed_img)
        all_features.append(features.flatten()) # Flatten the vector
    return np.array(all_features)

# --- 2. Load Images and Extract Features ---

# !IMPORTANT: Replace with the actual paths to your image files
# For this example, I'll simulate your 6 images.
# In a real scenario, you'd get these from a folder.
# image_paths = [
#     'path/to/sample1.jpg', 
#     'path/to/sample2.jpg',
#     'path/to/sample3.jpg',
#     'path/to/sample4.jpg',
#     'path/to/sample5.jpg',
#     'path/to/sample6.jpg'
# ]
# image_paths = [f"path/to/image_{i}.jpg" for i in range(1, 7)]
# Since I can't access your files, I'll use placeholder paths
# You MUST replace this list with your actual file paths.
image_paths = [
    "data/2023-027735#12#1_1.png", 
    "data/2023-027735#12#1_2.png", 
    "data/2023-027735#12#1_3.png", 
    "data/2023-027735#12#1_4.png", 
    "data/2023-027735#12#1_5.png", 
    "data/2023-027735#12#1_6.png"
]
# For the code to run, let's create dummy files (you can skip this)
print("Creating dummy image files for demonstration...")
for p in image_paths:
    if not os.path.exists(p):
        dummy_img = np.random.rand(224, 224, 3) * 255
        image.save_img(p, dummy_img)

# This is the key step:
# features_array.shape will be (6, 2048)
features_array = extract_features(image_paths)
print(f"Extracted features. Shape: {features_array.shape}")


# --- 3. Find Optimal Number of Clusters (k) ---

# We'll test k from 2 up to the (number of samples - 1)
# Your image has 6 samples, so we test k=2, 3, 4, 5
max_k = len(image_paths) - 1
silhouette_scores = {}

print(f"Finding best k (from 2 to {max_k})...")
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_array)
    
    # Calculate silhouette score
    score = silhouette_score(features_array, cluster_labels)
    silhouette_scores[k] = score
    print(f"  k={k}, Silhouette Score: {score:.4f}")

# Find the k with the highest silhouette score
best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nBest k (highest silhouette score) is: {best_k}")


# --- 4. Perform Final Clustering and Show Results ---

# Now, cluster with the best_k
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(features_array)

print("\n--- Clustering Results ---")
print(f"Found {best_k} distinct specimens (clusters).")

# Group images by their assigned cluster
clusters = {}
for i, label in enumerate(final_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(image_paths[i])

# Print the groups
for cluster_id, files in clusters.items():
    print(f"\nSpecimen (Cluster) {cluster_id}:")
    for file in files:
        print(f"  - {file}")
