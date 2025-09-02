import tensorflow as tf

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 32
DATA_DIR = '/mnt/harsh_project/archive(1)/PlantVillage'
VALIDATION_SPLIT = 0.2

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('plant_disease_model.keras')

# Create the validation dataset
print("Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Evaluate the model
print("Evaluating model...")
loss, accuracy = model.evaluate(val_ds)

print(f"\nModel Accuracy on the validation set: {accuracy * 100:.2f}%")
