import numpy as np
import cv2

def w2d(img, level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_float = np.float32(img_gray) / 255.0

    # Define a simple filter (e.g., Sobel, Laplacian, etc.)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])  # This is a simple edge-detection kernel (Laplacian)

    # Apply convolution (using filter2D for kernel application)
    coeffs = cv2.filter2D(img_float, -1, kernel)

    # Process and scale coefficients as needed
    processed_img = np.abs(coeffs)  # Use absolute value to simulate transformation
    processed_img *= 255
    return np.uint8(processed_img)

# Usage example:
img = cv2.imread('your_image.jpg')  # Replace with your actual image
processed_img = w2d(img)
cv2.imshow("Processed Image", processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
