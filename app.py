import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/uploads')
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), 'digit_recognition_model.h5')

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pre-trained model
model = load_model(app.config['MODEL_PATH'])

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.asarray(image, dtype='float32') / 255.0
    image_array = image_array.reshape((1, 28, 28, 1))
    return image_array

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return redirect(url_for('index'))
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return redirect(url_for('index'))
    
    predictions = []
    
    for file in files:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        image = Image.open(image_path)
        image_array = preprocess_image(image)
        
        pred = model.predict(image_array)
        predicted_digit = np.argmax(pred)
        
        predictions.append((file.filename, predicted_digit))
    
    return render_template('index.html', prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)
