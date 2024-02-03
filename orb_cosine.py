import cv2
import numpy as np
import os
from voyager import Index, Space
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path, descriptor_dim=128):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print(f"Error: Unable to read the image at '{image_path}'")
        return None

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    # Trim or resize the descriptors to the desired dimension
    descriptors = descriptors[:, :descriptor_dim].flatten()
    descriptors = descriptors.astype(np.float32)

    return keypoints, descriptors

def build_voyager_index_for_folder(folder_path, descriptor_dim=128):
    all_embeddings = []
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            keypoints, descriptors = preprocess_image(image_path, descriptor_dim)

            if descriptors is not None:
                # Ensure the descriptors are trimmed to the desired dimension
                descriptors = descriptors[:descriptor_dim]
                all_embeddings.append(descriptors)
                image_paths.append(image_path)

    if len(all_embeddings) == 0:
        print("Error: No images found in the folder.")
        return None

    embedding_dim = descriptor_dim
    space = Space.Cosine  # Use Cosine Similarity space
    voyager_index = Index(space, num_dimensions=embedding_dim)

    for vec in all_embeddings:
        voyager_index.add_item(vec)

    return voyager_index, image_paths

def find_nearest_neighbor(image_path, voyager_index, image_paths, threshold=0.3):
    query_keypoints, query_descriptors = preprocess_image(image_path, voyager_index.num_dimensions)

    if query_descriptors is not None:
        # Ensure the query descriptors are trimmed to the desired dimension
        query_vector = query_descriptors[:voyager_index.num_dimensions]

        # Using voyager's nearest neighbor search
        neighbors, _ = voyager_index.query(query_vector, k=len(image_paths))

        # Check if there are any results
        if len(neighbors) > 0:
            best_match_index = -1
            best_match_similarity = -1.0  # Set to -1 initially

            for i, index in enumerate(neighbors):
                index = int(index)

                # Compute the similarity between entire images (not individual embeddings)
                similarity = compute_image_similarity(image_path, image_paths[index])

                # Update best match if the similarity is higher
                if similarity > best_match_similarity:
                    best_match_similarity = similarity
                    best_match_index = index

            # You can customize the comparison criteria based on similarity or other factors
            if best_match_similarity >= threshold:
                print(f"\nBest Match:")
                print(f"  Neighbor: {image_paths[best_match_index]}")
                print(f"  Similarity: {best_match_similarity}")
                print(f"  Match found (above or equal to threshold)\n")
            else:
                print("\nNo match found (below threshold)\n")
        else:
            print("No results found.")

# Function to compute image similarity using structural similarity index (SSI)
def compute_image_similarity(query_image_path, neighbor_image_path):
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    neighbor_image = cv2.imread(neighbor_image_path, cv2.IMREAD_GRAYSCALE)

    if query_image is None or neighbor_image is None:
        print("Error: Unable to read one of the images.")
        return -1.0

    # Resize images to the same dimensions
    height, width = min(query_image.shape[0], neighbor_image.shape[0]), min(query_image.shape[1], neighbor_image.shape[1])
    query_image = cv2.resize(query_image, (width, height))
    neighbor_image = cv2.resize(neighbor_image, (width, height))

    # Compute structural similarity index
    similarity = ssim(query_image, neighbor_image)

    return similarity
