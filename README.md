# Final Macmillan Internship - Phavorite! Image Processing Model 

- uses Spotify's Voyager, an approximate nearest-neighbor search library for Python
- uses ORB, a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance
- imports openCV, os (operating system), scikit-image

Goals:
Given a Macmillan catalogue library with thousands of covers as a set of images with ISBNs as their titles, we are finding a way to efficient compare a user-input book cover with each image in the library to find the best match. We do this by using ORB to preprocess images by generating embeddings for each after cropping dimensions, then use Voyager to efficiently find the nearest neighbor match(es). We must find a mechanism to generate a set of embeddings that are robust even with rotations, distortions, occlusions, etc. by researching different feature-detection algorithms. 

