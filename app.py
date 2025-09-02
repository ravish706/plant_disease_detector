import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
DATA_DIR = '/mnt/harsh_project/archive(1)/PlantVillage'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get class names from the directory structure
class_names = sorted(os.listdir(DATA_DIR))

# Load the trained model
print("Loading model...")
model = load_model('plant_disease_model.keras')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        preds = model_predict(filepath, model)
        score = tf.nn.softmax(preds[0])
        
        class_name = class_names[np.argmax(score)]
        # Format the class name
        if '___' in class_name:
            plant_name, disease_name = class_name.split('___')
            plant_name = plant_name.replace('_', ' ')
            disease_name = disease_name.replace('_', ' ')
            formatted_prediction = f"{plant_name} - {disease_name}"
        else:
            parts = class_name.split('_')
            plant_name = parts[0]
            disease_name = " ".join(parts[1:])
            formatted_prediction = f"{plant_name} - {disease_name}"

        prediction = {
            "class": formatted_prediction,
            "confidence": f"{100 * np.max(score):.2f}"
        }
        
        return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
