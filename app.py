import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__),'digit_recognition_model.h5')

# Load the pre-trained model

# Load the pre-trained model
model = load_model(app.config['MODEL_PATH'])

# Function to preprocess image
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Convert image to numpy array and normalize
    image_array = np.asarray(image, dtype='float32') / 255.0
    # Reshape to match CNN input shape
    image_array = image_array.reshape((1, 28, 28, 1))
    return image_array

# Route to home page
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save the uploaded file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Load and preprocess the image
    image = Image.open(image_path)
    image_array = preprocess_image(image)
    
    # Use the model to predict the digit
    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions)
    
    # Delete the uploaded image file
    os.remove(image_path)
    
    return render_template('index.html', prediction=predicted_digit, image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
