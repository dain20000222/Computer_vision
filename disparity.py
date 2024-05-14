import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# Constants for the streo imagery
WINDOW_NAME = "Disparity"
MAX_DISPARITIES = 64  
MIN_BLOCK_SIZE = 5   
FOCAL_LENGTH = 5806.559  
BASELINE = 174.019       
DISPARITY_OFFSET = 114.291 

# Calculate disparity map from stereo images
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image

# Calculate depth map from disparity map
def calculate_depth_map(baseline, focal_length, disparity, disparity_offset):
    depth_map = np.ones_like(disparity)
    depth_map *= focal_length * baseline
    depth_map /= disparity + np.ones_like(disparity) * disparity_offset
    return depth_map

# Dynamically update disparity settings upon trackbar changes
def change_disparity(value):
    global MAX_DISPARITIES
    MAX_DISPARITIES = (value // 16) * 16
    disparity = getDisparityMap(imgL, imgR, MAX_DISPARITIES, MIN_BLOCK_SIZE)
    normalized_disparity = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow(WINDOW_NAME, normalized_disparity)

# Dynamically update block size settings upon trackbar changes
def change_block_size(value):
    global MIN_BLOCK_SIZE
    MIN_BLOCK_SIZE = value - 1 if value % 2 == 0 else value
    MIN_BLOCK_SIZE = max(MIN_BLOCK_SIZE, 5)
    disparity = getDisparityMap(imgL, imgR, MAX_DISPARITIES, MIN_BLOCK_SIZE)
    normalized_disparity = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow(WINDOW_NAME, normalized_disparity)

# Visualize depth
def visualize_depth(depth):
    x = []
    y = []
    z = []
    for r in range(np.shape(depth)[0]):
        for c in range(np.shape(depth)[1]):
            # Thresholding
            if (depth[r][c] < 7600):
                x += [depth[r,c] * (c / FOCAL_LENGTH) - BASELINE / 2]
                y += [depth[r,c] * (r / FOCAL_LENGTH)]
                z += [depth[r, c]]

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, z, y, 'green', s=0.5)

    # 3D view
    # ax.view_init(20, -125)
    # Top view
    # ax.view_init(90, -90) 
    # Side view
    # ax.view_init(0, 0)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.show()

if __name__ == "__main__":
    # Load and process images
    imgL = cv2.imread("umbrellaL.png", cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread("umbrellaR.png", cv2.IMREAD_GRAYSCALE)
    
    imgL = cv2.Canny(cv2.GaussianBlur(imgL, (5, 5), 1), 45, 90)
    imgR = cv2.Canny(cv2.GaussianBlur(imgR, (5, 5), 1), 45, 90)

    # Create a window to display the image
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Disparities", WINDOW_NAME, MAX_DISPARITIES, 256, change_disparity)
    cv2.createTrackbar("Block Size", WINDOW_NAME, MIN_BLOCK_SIZE, 256, change_block_size)
    
    # Initial disparity and depth map calculation and display
    disparity = getDisparityMap(imgL, imgR, MAX_DISPARITIES, MIN_BLOCK_SIZE)
    normalized_disparity = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    cv2.imshow(WINDOW_NAME, normalized_disparity)
    depth = calculate_depth_map(BASELINE, FOCAL_LENGTH, disparity, DISPARITY_OFFSET)
    visualize_depth(depth)
    
    while True:
        key = cv2.waitKey(1)
        if key in [ord(" "), 27]:  
            break
    cv2.destroyAllWindows()
