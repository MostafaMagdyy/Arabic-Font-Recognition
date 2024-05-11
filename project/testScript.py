import os
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import accuracy_score
import skimage.io as io



def unsharp_masking(image, blur_radius=5, sharpen_amount=1.0):
    """Apply unsharp masking followed by dilation to enhance image details."""
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), blur_radius)
    
    # Apply unsharp masking
    sharpened = cv2.addWeighted(image, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    
    # Ensure the sharpened image doesn't have values below the original image
    sharpened = np.where(image >= blurred, sharpened, image)
    
    return sharpened

def remove_noise22(image, dilation_kernel=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_filtered = cv2.bilateralFilter(gray, 15, sigmaColor=3, sigmaSpace=10)
    
    median_blurred = cv2.medianBlur(bilateral_filtered, 3)
    result = unsharp_masking(median_blurred, blur_radius=10, sharpen_amount=4) * 255
    
    result = np.uint8(result)
    edges = cv2.Canny(result, 80, 255,apertureSize=3)
       # Optionally apply dilation
    if dilation_kernel is not None:
        # Perform dilation
        edges = cv2.dilate(edges, dilation_kernel)
    return edges

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return descriptors

def test(directory, kmeans_model_filename, svm_model_filename):
    # Load the KMeans model
    print(os.getcwd())
    with open(kmeans_model_filename, 'rb') as file:
        loaded_kmeans_model = pickle.load(file)
    
    # Load the SVM model
    with open(svm_model_filename, 'rb') as file:
        loaded_svm_model = pickle.load(file)

    test_histograms = []
    
    # Iterate over sorted files in the directory
    file_list = sorted(os.listdir(directory))
    for file_name in file_list:
        input_image_path = os.path.join(directory, file_name)
        
        # Apply preprocessing (remove noise)
        image = io.imread(input_image_path)
        preprocessed_image = remove_noise22(image,dilation_kernel=None)
        cv2.imshow("one",preprocessed_image)
        descriptors = extract_sift_features(preprocessed_image)
        if descriptors is not None and descriptors.shape[0] > 0 and descriptors.shape[1] > 0:
            nearest_clusters = pairwise_distances_argmin_min(descriptors, loaded_kmeans_model.cluster_centers_)[0]
            histogram, _ = np.histogram(nearest_clusters, bins=np.arange(loaded_kmeans_model.n_clusters + 1))
            test_histograms.append((histogram, file_name))  # Assuming filename is the label
    
    X_test = [histogram for histogram, _ in test_histograms]
    y_test = [label for _, label in test_histograms]
    
    y_pred = loaded_svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

directory = "tests"
kmeans_model_filename = "kmeans_model.pkl"
svm_model_filename = "svm_model.pkl"
test(directory, kmeans_model_filename, svm_model_filename)
