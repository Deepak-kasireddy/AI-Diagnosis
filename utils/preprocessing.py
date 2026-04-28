import cv2
import numpy as np
from PIL import Image

def preprocess_xray(image_path, target_size=(224, 224)):
    """
    Preprocess X-ray images: Grayscale, CLAHE, Resize, Normalize.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    
    # Resize
    img_resized = cv2.resize(img_clahe, target_size)
    
    # Normalize to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Add channel dimension
    img_norm = np.expand_dims(img_norm, axis=-1)
    
    return img_norm

def preprocess_blood_smear(image_path, target_size=(224, 224)):
    """
    Preprocess Blood Smear images: Color normalization (simplified), Resize, Normalize.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Normalize to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0
    
    return img_norm

def get_pil_image(image_path):
    return Image.open(image_path)
