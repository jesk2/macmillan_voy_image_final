import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
from voyager import Space, Index, StorageDataType

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read the image at '{image_path}'")
        return None

    # Resize the image if needed
    img_resized = cv2.resize(img, (64, 128))  # Adjust the size as needed

    return img_resized

def build_voyager_index_for_folder(folder_path):
    all_embeddings = []
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = preprocess_image(image_path)

            if img is not None:
                hog_descriptor = compute_hog_descriptor(img)
                all_embeddings.append(hog_descriptor)
                image_paths.append(image_path)

    if len(all_embeddings) == 0:
        print("Error: No images found in the folder.")
        return None, None

    dimensions = len(all_embeddings[0])
    space = Space.Euclidean
    storage_data_type = StorageDataType.Float32
    voyager_index = Index(space, dimensions, storage_data_type=storage_data_type)

    for idx, vec in enumerate(all_embeddings):
        voyager_index.add_items([vec], [idx])

    return voyager_index, image_paths

def find_nearest_neighbor(image_path, voyager_index, image_paths, threshold=1000):
    query_img = preprocess_image(image_path)

    if query_img is not None:
        query_descriptor = compute_hog_descriptor(query_img)

        best_distance = float('inf')
        best_match_index = -1

        for idx in range(len(voyager_index)):
            ref_descriptor = voyager_index.get_vectors([idx])[0]

            # Calculate Euclidean distance
            euclidean_distance = euclidean_distances([query_descriptor], [ref_descriptor])[0][0]

            print(f"Comparison {idx + 1}:")
            print(f"  Neighbor: {image_paths[idx]}")
            print(f"  Euclidean Distance: {euclidean_distance}")

            if euclidean_distance < best_distance:
                best_distance = euclidean_distance
                best_match_index = idx

        if best_distance < threshold:
            print(f"\nQuery Image: {image_path}")
            print(f"Best Match Found: {image_paths[best_match_index]}")
        else:
            print("No match found.")

def compute_hog_descriptor(image):
    # Calculate HOG descriptor
    win_size = (64, 128)  # Adjust the window size as needed
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_descriptor = hog.compute(image)
    return hog_descriptor.flatten()
