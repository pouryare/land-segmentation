import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('Land-Segmentation.keras')

def preprocess_image(image_path, shape=(128, 128)):
    img = plt.imread(image_path)
    img = cv2.resize(img, shape)
    return np.expand_dims(img, axis=0)

def predict(image):
    prediction = model.predict(image)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(os.getcwd(), 'uploads', filename)
        file.save(file_path)
        
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        prediction = predict(processed_image)
        
        # Apply sharpening filter
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_prediction = cv2.filter2D(prediction, -1, sharpen_kernel)
        
        # Ensure prediction values are in the range [0, 1]
        sharpened_prediction = np.clip(sharpened_prediction, 0, 1)
        
        # Save the prediction image
        output_filename = f"prediction_{filename}"
        output_path = os.path.join(os.getcwd(), 'static', output_filename)
        plt.imsave(output_path, sharpened_prediction)
        
        return jsonify({'success': True, 'prediction_url': f'/static/{output_filename}'})

if __name__ == "__main__":
    os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'static'), exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)