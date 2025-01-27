import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load the saved model
model_save_path = 'blood_group_model.h5'
model = load_model(model_save_path)
print("Model loaded successfully.")

# Image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Class indices (use the same as in training)
class_indices = {
    0: 'A+', 
    1: 'A-', 
    2: 'AB+', 
    3: 'AB-', 
    4: 'B+', 
    5: 'B-', 
    6: 'O+', 
    7: 'O-'  
}

# Path to the test image
test_image_path = 'E:\Python Projects\Fingerprint_blood_group\dataset_blood_group\O+\cluster_6_59.BMP'

# Preprocess the test image
test_image = load_img(test_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Resize image
test_image_array = img_to_array(test_image)  # Convert to array
test_image_array = test_image_array / 255.0  # Normalize pixel values
test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(test_image_array)
predicted_class_index = np.argmax(predictions)  # Get index of the highest probability
predicted_class_label = class_indices[predicted_class_index]

print(f"Predicted Blood Group: {predicted_class_label}")
