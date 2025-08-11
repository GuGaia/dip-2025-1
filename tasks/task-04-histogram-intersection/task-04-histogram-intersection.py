import numpy as np
import cv2

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    histogramImg1, _ = np.histogram(img1, bins=256, range=(0, 255))
    histogramImg2, _ = np.histogram(img2, bins=256, range=(0, 255))

    #Normalizando valores
    histogramImg1 = histogramImg1.astype(float)/ histogramImg1.sum()
    histogramImg2 = histogramImg2.astype(float)/ histogramImg2.sum()

    intersection = np.sum(np.minimum(histogramImg1, histogramImg2))

    return float(intersection)
