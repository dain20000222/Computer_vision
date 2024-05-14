import numpy as np
import cv2

# Constants for the selective focus
WINDOW_NAME = "Disparity"
WINDOW_DEPTH = "Depth Approximation"
WINDOW_FOCUS = "Selective Focus"
MAX_DISPARITIES = 64  
MIN_BLOCK_SIZE = 5   
DISPARITIES = 16
BLOCK_SIZE = 23
K = 2
THRESL = 31
THRESH = 100
PROCESSING_MODE = 2

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

# Approximate the depth from the disparity
def approximate_depth(disparity, k):
    depth = 1.0 / (disparity + k)
    return depth

# Change K
def change_k(val):
    global k, depth
    k = val
    depth = approximate_depth(disparity, k)
    normalized_depth = np.interp(depth, (depth.min(), depth.max()), (0.0, 1.0))
    cv2.imshow(WINDOW_DEPTH, normalized_depth)

# Change threshold lower bound
def change_thresL(val):
    global thresL
    thresL = val
    if PROCESSING_MODE == 2:
        output_image = mono_background(imgLColor, depth, thresL, thresH)
    elif PROCESSING_MODE == 1:
        output_image = blur_background(imgLColor, depth, thresL, thresH)
    else:
        output_image = blur_background(imgL, depth, thresL, thresH)
    cv2.imshow(WINDOW_FOCUS, output_image)

# Change threshold upper bound
def change_thresH(val):
    global thresH
    thresH = val
    if PROCESSING_MODE == 2:
        output_image = mono_background(imgLColor, depth, thresL, thresH)
    elif PROCESSING_MODE == 1:
        output_image = blur_background(imgLColor, depth, thresL, thresH)
    else:
        output_image = blur_background(imgL, depth, thresL, thresH)
    cv2.imshow(WINDOW_FOCUS, output_image)

# Apply blur to the background
def blur_background(img, depth, thresL, thresH):
    blurred = cv2.blur(img, (10,10))
    out = np.copy(img)
    for r in range(np.shape(depth)[0]):
        for c in range(np.shape(depth)[1]):
            if(depth[r][c] >= float(thresL)/100 and depth[r][c] <= float(thresH)/100):
                out[r][c] = blurred[r][c]
    return out

# Convert background to monochrome
def mono_background(img, depth, thresL, thresH):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.copy(img)
    for r in range(np.shape(depth)[0]):
        for c in range(np.shape(depth)[1]):
            if(depth[r][c] >= float(thresL)/100 and depth[r][c] <= float(thresH)/100):
                out[r][c][0] = grayscale[r][c]
                out[r][c][1] = grayscale[r][c]
                out[r][c][2] = grayscale[r][c]
    return out 

if __name__ == "__main__":
    # Load images
    imgL = cv2.imread("girlL.png", cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread("girlR.png", cv2.IMREAD_GRAYSCALE)
    imgLColor = cv2.imread("girlL.png", cv2.IMREAD_COLOR)
    imgRColor = cv2.imread("girlR.png", cv2.IMREAD_COLOR)

    # Initialize disparity and depth calculations
    disparity = getDisparityMap(imgL, imgR, DISPARITIES, BLOCK_SIZE)
    depth = approximate_depth(disparity, K)

    # Initialize windows
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(WINDOW_DEPTH)
    cv2.namedWindow(WINDOW_FOCUS)

    # Create trackbars for each window
    cv2.createTrackbar("Disparities", WINDOW_NAME, DISPARITIES, 256, change_disparity)
    cv2.createTrackbar("Block Size", WINDOW_NAME, BLOCK_SIZE, 256, change_block_size)
    
    cv2.createTrackbar("K Value", WINDOW_DEPTH, K, 256, change_k)
    
    cv2.createTrackbar("Threshold L", WINDOW_FOCUS, THRESL, 256, change_thresL)
    cv2.createTrackbar("Threshold H", WINDOW_FOCUS, THRESH, 256, change_thresH)

    cv2.waitKey(0)
    cv2.destroyAllWindows()