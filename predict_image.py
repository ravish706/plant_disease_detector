import tensorflow as tf
import numpy as np
import os
import sys

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
DATA_DIR = '/mnt/harsh_project/archive(1)/PlantVillage'

# Get class names from the directory structure
class_names = sorted(os.listdir(DATA_DIR))

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('plant_disease_model.keras')

# Get image path from command line argument or use a default
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = '/mnt/harsh_project/archive(1)/PlantVillage/Pepper__bell___healthy/00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL 7886.JPG'

# Load and preprocess the image
print(f"Loading image: {image_path}")
try:
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Make a prediction
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Print the result
print(
    "\nThis image most likely belongs to the class '{}' with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
