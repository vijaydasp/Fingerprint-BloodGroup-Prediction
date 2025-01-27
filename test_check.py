import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

IMG_HEIGHT = 64
IMG_WIDTH = 64

# Load the model
loaded_model = load_model('blood_group_model.h5')

# Test the model with an image
def test_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    predictions = loaded_model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    
    # Return the predicted class
    return predicted_class

# Test the model with a new image
image_path = 'E:\Python Projects\Fingerprint_blood_group\dataset_blood_group\A+\cluster_0_116.BMP'  # Replace with the actual image path
predicted_class = test_image(image_path)
print(f'Predicted Class: {predicted_class}')
