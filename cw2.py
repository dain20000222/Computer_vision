import cv2
import numpy as np
import os
from scipy import ndimage, spatial
import matplotlib.pyplot as plt
import sys

# Function for Harris Point Detector
def HarrisPointsDetector(image, threshold_percentage=0.05):
    # Normalize the image to range [0, 1]
    normalized_image = image.astype(np.float64) / 255.0
    
    # Compute the gradient along the x and y axes using Sobel operator
    gradient_x = ndimage.sobel(normalized_image, axis=1, mode="reflect")
    gradient_y = ndimage.sobel(normalized_image, axis=0, mode="reflect")
    
    # Calculate gradient orientations in degrees
    orientations = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    
    # Compute the square of gradients
    gradient_x_squared = np.square(gradient_x)
    gradient_y_squared = np.square(gradient_y)
    gradient_xy = np.multiply(gradient_x, gradient_y)
    
    # Apply 5x5 Gaussian filter with 0.5 sigma
    gradient_x_squared_smoothed = ndimage.gaussian_filter(gradient_x_squared, sigma=0.5, mode="reflect", radius=2)
    gradient_y_squared_smoothed = ndimage.gaussian_filter(gradient_y_squared, sigma=0.5, mode="reflect", radius=2)
    gradient_xy_smoothed = ndimage.gaussian_filter(gradient_xy, sigma=0.5, mode="reflect", radius=2)
    
    # Calculate determinant and trace for Harris response
    determinant = np.multiply(gradient_x_squared_smoothed, gradient_y_squared_smoothed) - np.square(gradient_xy_smoothed)
    trace = gradient_x_squared_smoothed + gradient_y_squared_smoothed
    
    # Harris corner response calculation
    R = determinant - (0.05 * np.square(trace))
    
    # Find local maxima in a 7x7 neighbourhood, using reflective padding
    local_maxima = ndimage.maximum_filter(R, size=7, mode="reflect") == R
    
    # Collect keypoints where corner response is above threshold
    keypoints = []
    rows, cols = normalized_image.shape
    threshold = np.max(R) * threshold_percentage
    for y in range(rows):
        for x in range(cols):
            if local_maxima[y, x] and R[y, x] >= threshold:
                keypoints.append(cv2.KeyPoint(x, y,7, orientations[y, x], R[y, x]))
    
    return keypoints

# Fuction to match features based on the SSD distance
def SSDFeatureMatcher(descriptors1, descriptors2):
    # Check if either of the descriptor lists is empty, return empty list if true.
    if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        return []

    # Compute the square Euclidean distance between each pair of the two collections of inputs.
    distances = spatial.distance.cdist(descriptors1, descriptors2, 'sqeuclidean')

    # For each descriptor in descriptors1, find the best match in descriptors2 based on the SSD distance.
    matches = [cv2.DMatch(i, np.argmin(distances[i]), np.amin(distances[i]))
               for i in range(len(distances))]
    
    return matches

# Fuction to matche features based on the ratio test
def RatioFeatureMatcher(descriptors1, descriptors2, ratio_threshold=0.75):
    # Check if either of the descriptor lists is empty, return empty list if true.
    if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        return []

    # Compute the square Euclidean distance between each pair of the two collections of inputs.
    distances = spatial.distance.cdist(descriptors1, descriptors2, 'sqeuclidean')
    matches = []
    # Apply the ratio test
    for i, row in enumerate(distances):
        # Sort the distances for each descriptor.
        sorted_indices = np.argsort(row)
        # If the best match distance is less than ratio_threshold times the second best, it's a good match.
        if row[sorted_indices[0]] < ratio_threshold * row[sorted_indices[1]]:
            matches.append(cv2.DMatch(i, sorted_indices[0], row[sorted_indices[0]]))
    
    return matches


if __name__ == "__main__":
    # Load the reference image and convert it to grayscale.
    reference_image = cv2.imread("bernieSanders.jpg", cv2.IMREAD_COLOR)
    grayscale_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detectors with different scoring functions.
    orb_detector_harris = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
    orb_detector_fast = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

    # Detect keypoints using ORB with Harris and FAST scores respectively.
    keypoints_harris = orb_detector_harris.detect(grayscale_image, None)
    keypoints_fast = orb_detector_fast.detect(grayscale_image, None)

    # Compute descriptors for keypoints detected by both ORB detectors.
    keypoints_harris, descriptors_harris = orb_detector_harris.compute(grayscale_image, keypoints_harris)
    keypoints_fast, descriptors_fast = orb_detector_fast.compute(grayscale_image, keypoints_fast)

    # Draw keypoints detected by each method on the grayscale image.
    image_with_harris_keypoints = cv2.drawKeypoints(grayscale_image, keypoints_harris, None, color=(0, 255, 0), flags=0)
    image_with_fast_keypoints = cv2.drawKeypoints(grayscale_image, keypoints_fast, None, color=(0, 255, 0), flags=0)

    # Save images
    cv2.imwrite("orb_harris_keypoints.jpg", image_with_harris_keypoints.astype(np.uint8))
    cv2.imwrite("orb_fast_keypoints.jpg", image_with_fast_keypoints.astype(np.uint8))

    # Apply Gaussian blur to the grayscale image.
    blurred_grayscale_image = ndimage.gaussian_filter(grayscale_image, sigma=2, mode="reflect", radius=2)

    # Detect keypoints using a custom Harris point detector function on the blurred image.
    keypoints_custom_harris = HarrisPointsDetector(blurred_grayscale_image)

    # Compute descriptors for the custom Harris keypoints.
    _, descriptors_custom_harris = orb_detector_harris.compute(blurred_grayscale_image, keypoints_custom_harris)

    # Draw the custom Harris keypoints on the original grayscale image.
    image_with_custom_harris_keypoints = cv2.drawKeypoints(grayscale_image, keypoints_custom_harris, None, color=(0, 255, 0), flags=0)

    # Save the image with custom Harris keypoints to a file.
    cv2.imwrite("harris_keypoints.jpg", image_with_custom_harris_keypoints.astype(np.uint8))
    
    # Print the number of detected keypoints for a quantitative comparison.
    print("Keypoints detected - Harris: ", len(keypoints_harris), ", FAST: ", len(keypoints_fast), ", Custom Harris: ", len(keypoints_custom_harris))

    # Plot the number of keypoints detected as a function of the threshold value.
    threshold_coefficients = [0.005 * i for i in range(1, 21)]
    keypoints_counts = [len(HarrisPointsDetector(blurred_grayscale_image, threshold)) for threshold in threshold_coefficients]
    plt.plot(threshold_coefficients, keypoints_counts)
    plt.xlabel('Threshold Coefficient')
    plt.ylabel('Number of Keypoints')
    plt.title("Number of Keypoints vs. Threshold Coefficient")
    plt.show()

    # Process resource images
    for filename in os.listdir("resources"):
        # Read and preprocess the image.
        image_path = os.path.join("resources", filename)
        sample_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        sample_image_grayscale = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        sample_image_blurred = ndimage.gaussian_filter(sample_image_grayscale, sigma=2, mode="reflect", radius=2)

        # Detect keypoints using the custom Harris detector.
        keypoints_sample = HarrisPointsDetector(sample_image_blurred)

        # Draw keypoints on the blurred sample image.
        sample_image_with_keypoints = cv2.drawKeypoints(sample_image_blurred, keypoints_sample, None, color=(0, 255, 0), flags=0)

        # Compute descriptors for the detected keypoints.
        _, descriptors_sample = orb_detector_harris.compute(sample_image_blurred, keypoints_sample)

        # Match features between the reference and the current sample using SSD.
        matches_ssd = sorted(SSDFeatureMatcher(descriptors_custom_harris, descriptors_sample), key=lambda x: x.distance)
        matched_image_ssd = cv2.drawMatches(image_with_custom_harris_keypoints, keypoints_custom_harris, sample_image_with_keypoints, keypoints_sample, matches_ssd, None, flags=2)

        # Save the image with SSD matched features.
        output_filename_ssd = filename.split(".")[0] + "_SSD.jpg"
        output_path = os.path.join("result", output_filename_ssd)
        cv2.imwrite(output_path, matched_image_ssd)

        # Match features using the Ratio Feature Matcher
        matches_rfm = sorted(RatioFeatureMatcher(descriptors_custom_harris, descriptors_sample), key=lambda x: x.distance)
        matched_image_rfm = cv2.drawMatches(image_with_custom_harris_keypoints, keypoints_custom_harris, sample_image_with_keypoints, keypoints_sample, matches_rfm, None, flags=2)

        # Save the image with RFM matched features.
        output_filename_rfm = filename.split(".")[0] + "_RFM.jpg"
        output_path = os.path.join("result", output_filename_rfm)
        cv2.imwrite(output_path, matched_image_rfm)