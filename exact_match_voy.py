import cv2
import numpy as np
import os
from voyager import Index, Space, StorageDataType

def preprocess_image(image_path, descriptor_dim=128):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print(f"Error: Unable to read the image at '{image_path}'")
        return None

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    descriptors = descriptors[:, :descriptor_dim]
    descriptors = descriptors.astype(np.float32)

    return descriptors

def build_voyager_index_for_folder(folder_path, descriptor_dim=128):
    all_embeddings = []
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            descriptors = preprocess_image(image_path, descriptor_dim)

            if descriptors is not None:
                all_embeddings.append(descriptors.flatten())
                image_paths.append(image_path)

    if len(all_embeddings) == 0:
        print("Error: No images found in the folder.")
        return None, None

    embedding_dim = all_embeddings[0].shape[0]
    space = Space.Euclidean  # You may choose a different space if needed
    dimensions = embedding_dim
    storage_data_type = StorageDataType.Float32
    voyager_index = Index(space, dimensions, storage_data_type=storage_data_type)

    for idx, vec in enumerate(all_embeddings):
        voyager_index.add_items([vec], [idx])

    return voyager_index, image_paths

def find_nearest_neighbor(image_path, voyager_index, image_paths, threshold=0.5):
    query_descriptors = preprocess_image(image_path)

    if query_descriptors is not None:
        query_vector = query_descriptors.flatten()

        best_euclidean_distance = float('inf')
        best_match_index = -1

        for idx in range(len(voyager_index)):
            ref_vector = voyager_index.get_vectors([idx])[0]
            euclidean_distance = np.linalg.norm(query_vector - ref_vector)

            print(f"Comparison {idx + 1}:")
            print(f"  Neighbor: {image_paths[idx]}")
            print(f"  Euclidean Distance: {euclidean_distance}")

            if euclidean_distance < best_euclidean_distance:
                best_euclidean_distance = euclidean_distance
                best_match_index = idx

        euclidean_distance_threshold = 15000

        if best_euclidean_distance < euclidean_distance_threshold:
            print(f"\nQuery Image: {image_path}")
            print(f"Best Match Found: {image_paths[best_match_index]}")
        else:
            print("No match found.")
