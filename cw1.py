import cv2
import numpy as np
from matplotlib import pyplot as plt

def convolve(image, kernel):
    """
    Apply a 2D convolution between the given image and kernel.
    This function pads the image with zeros, performs the convolution,
    and returns the filtered image.
    
    Parameters:
    - image: Input image as a 2D numpy array.
    - kernel: Convolution kernel as a 2D numpy array.
    
    Returns:
    - Filtered image as a 2D numpy array.
    """
    image = image.astype(np.float64)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y+3, x:x+3]
            output[y, x] = np.sum(region * kernel)
    return output

def threshold(image, threshold_value, invert=False):
    """
    Apply binary thresholding to an image. Pixels below the threshold are set to 0,
    and pixels above the threshold are set to 255. Optionally invert the thresholded image.
    
    Parameters:
    - image: Input image as a 2D numpy array.
    - threshold_value: Threshold value as an integer.
    - invert: Boolean flag to invert the thresholded image.
    
    Returns:
    - Thresholded image as a 2D numpy array.
    """
    output = np.copy(image)
    output[output <= threshold_value] = 0
    output[output > threshold_value] = 255
    if invert:
        output = 255 - output
    return output

# Load the input image in grayscale
image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)

# Define edge detection filters: Prewitt and Sobel
prewitt_horizontal = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])

prewitt_vertical = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

sobel_horizontal = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

sobel_vertical = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

# Define Gaussian smoothing filters with different standard deviations
gaussian_kernel_std1 = np.array([[0.08, 0.12, 0.08],
                                 [0.12, 0.20, 0.12],
                                 [0.08, 0.12, 0.08]])

# Normalizing the Gaussian kernel for std2 to ensure its sum equals 1
gaussian_kernel_std2 = gaussian_kernel_std1 / np.sum(gaussian_kernel_std1)

# Define a 3x3 Mean filter and normalize it
mean_filter = np.array([[0.1, 0.0, 0.1],
                        [0.0, 0.1, 0.0],
                        [0.1, 0.0, 0.1]])
mean_filter = mean_filter / np.sum(mean_filter)

# Apply Gaussian smoothing before edge detection
gaussian_blur = True
if gaussian_blur:
    image = convolve(image, mean_filter)

# Edge detection using Prewitt filters
# horizontal_edges = convolve(image, prewitt_horizontal)
# vertical_edges = convolve(image, prewitt_vertical)

# Edge detection using Sobel filters
horizontal_edges = convolve(image, sobel_horizontal)
vertical_edges = convolve(image, sobel_vertical)

# Compute edge magnitude
edge_magnitude = np.sqrt(np.square(horizontal_edges) + np.square(vertical_edges))

# Normalize and scale edge magnitude to 0-255
horizontal_edges = cv2.convertScaleAbs(horizontal_edges)
vertical_edges = cv2.convertScaleAbs(vertical_edges)

edge_magnitude = edge_magnitude / np.amax(edge_magnitude[1:-1, 1:-1]) * 255
edge_magnitude = edge_magnitude.astype(np.uint8)

# Apply binary thresholding to the edge magnitude
threshold_value = 80 if gaussian_blur else 90
thresholded_edges = threshold(edge_magnitude, threshold_value)

# Display and save results
image = cv2.convertScaleAbs(image)
cv2.imwrite('Horizontal Edges.jpg', horizontal_edges)
cv2.imwrite('Vertical Edges.jpg', vertical_edges)
cv2.imwrite('Edge Magnitude.jpg', edge_magnitude)
cv2.imwrite('Thresholded Edges.jpg', thresholded_edges)

# Display the histogram of the edge magnitude image
plt.hist(edge_magnitude.ravel(), 256, [0, 256])
plt.title('Histogram of Edge Magnitude')
plt.show()

cv2.waitKey() 
cv2.destroyAllWindows()



