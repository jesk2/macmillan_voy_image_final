import cv2
import numpy as np
from annoy import AnnoyIndex
import os

# Function to preprocess an image and extract descriptors
def preprocess_image(image_path, descriptor_dim=128):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print(f"Error: Unable to read the image at '{image_path}'")
        return None

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    # Ensure the descriptors have the desired dimension
    descriptors = descriptors[:, :descriptor_dim]

    # Convert descriptors to numpy.float32
    descriptors = descriptors.astype(np.float32)

    return descriptors

# Function to build Annoy index for a folder of images
def build_annoy_index_for_folder(folder_path, descriptor_dim=128):
    all_embeddings = []
    image_paths = []

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            descriptors = preprocess_image(image_path, descriptor_dim)

            if descriptors is not None:
                all_embeddings.append(descriptors.flatten())
                image_paths.append(image_path)

    embedding_dim = all_embeddings[0].shape[0]  # Assuming all descriptors have the same dimension
    annoy_index = AnnoyIndex(embedding_dim, metric='euclidean')

    for idx, vec in enumerate(all_embeddings):
        annoy_index.add_item(idx, vec)

    num_trees = 50
    annoy_index.build(num_trees)

    return annoy_index, image_paths


# Function to find the nearest neighbor for a given image
def find_nearest_neighbor(image_path, annoy_index, image_paths):
    query_descriptors = preprocess_image(image_path)

    if query_descriptors is not None:
        query_vector = query_descriptors.flatten()

        # Initialize variables for the nearest neighbor
        nearest_distance = float('inf')
        nearest_index = -1

        # Iterate through all images to find the nearest neighbor
        for idx in range(annoy_index.get_n_items()):
            ref_vector = annoy_index.get_item_vector(idx)
            distance = np.linalg.norm(query_vector - ref_vector)

            print(f"Comparison {idx + 1}:")
            print(f"  Neighbor: {image_paths[idx]}")
            print(f"  Distance: {distance}")

            # Update nearest neighbor if the current distance is smaller
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = idx

        # Print information for the nearest neighbor
        print(f"\nQuery Image: {image_path}")
        print(f"Nearest Neighbor: {image_paths[nearest_index]}")
        print(f"Distance: {nearest_distance}")

        # Load and display the images
        cv2.imshow('Query Image', cv2.imread(image_path))
        nearest_neighbor_img = cv2.imread(image_paths[nearest_index])
        cv2.imshow('Nearest Neighbor', nearest_neighbor_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
folder_path = '/Users/jessica.kim/Desktop/macmillanCovers/'
query_image_path = '/Users/jessica.kim/Desktop/9781848130913.jpg'

annoy_index, image_paths = build_annoy_index_for_folder(folder_path)
find_nearest_neighbor(query_image_path, annoy_index, image_paths)
