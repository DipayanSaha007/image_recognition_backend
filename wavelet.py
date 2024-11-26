import numpy as np
import scipy.signal as signal
import cv2

def w2d(img, level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_float = np.float32(img_gray) / 255.0

    # Apply a simple wavelet transform using scipy (using DWT from signal)
    coeffs = signal.cwt(img_float, signal.ricker, np.arange(1, level + 1))

    # Process and scale coefficients as needed
    processed_img = np.abs(coeffs).mean(axis=0)  # Example processing
    processed_img *= 255
    return np.uint8(processed_img)
