import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64
CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Load the model
loaded_model = load_model('blood_group_model.h5')

def test_image(image_path, save_path='prediction_result.png'):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = loaded_model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_class_index]
    confidence_score = predictions[predicted_class_index]

    # Plotting
    plt.figure(figsize=(10, 4))

    # Input image with prediction
    plt.subplot(1, 2, 1)
    plt.imshow(load_img(image_path))
    plt.title(f'Predicted: {predicted_label}\nConfidence: {confidence_score:.2f}')
    plt.axis('off')

    # Bar chart for confidence
    plt.subplot(1, 2, 2)
    bars = plt.bar(CLASS_LABELS, predictions, color='skyblue')
    bars[predicted_class_index].set_color('green')
    plt.xticks(rotation=45)
    plt.title('Prediction Confidence')
    plt.ylabel('Confidence')
    plt.ylim([0, 1])

    # Annotate each bar
    for i, value in enumerate(predictions):
        plt.text(i, value + 0.01, f'{value:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()  # Close plot to free memory

    return predicted_label, confidence_score

# Run the test
image_path = 'D:\python projects\Fingerprint-BloodGroup-Prediction-main\dataset_blood_group\AB+\cluster_4_37.BMP'
predicted_label, confidence_score = test_image(image_path)
print(f'Predicted Blood Group: {predicted_label} (Confidence: {confidence_score:.2f})')
print("Prediction result saved as 'prediction_result.png'")
