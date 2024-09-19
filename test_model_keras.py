import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_face_detection.h5')

# Define class names and confidence threshold
class_names = ['dika', 'syiar']
confidence_threshold = 0.5  # Adjust as needed

# Define a function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Define a function to make predictions and get bounding boxes
def make_prediction(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    # For simplicity, assuming predictions include bounding boxes
    # This part of the code will depend on the specific output format of your model
    return predictions

# Define a function to display the results with bounding boxes
def display_results(img_path, predictions):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assuming predictions contain bounding boxes and class info
    # Example format: [class_id, confidence, x_min, y_min, x_max, y_max]
    for prediction in predictions:
        class_id, confidence, x_min, y_min, x_max, y_max = prediction
        if confidence < confidence_threshold:
            class_name = 'mencurigakan'
        else:
            class_name = class_names[int(class_id)]

        # Draw bounding box
        color = (0, 255, 0)  # Green for bounding boxes
        thickness = 2
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        cv2.putText(img, f'{class_name} ({confidence:.2f})', (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to perform detection
def detect_face(img_path):
    predictions = make_prediction(img_path)
    display_results(img_path, predictions)

# Example usage
image_path = 'syiar.jpg'  # Replace with the path to your test image
detect_face(image_path)
