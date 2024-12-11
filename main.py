from flask import Flask, request, jsonify, render_template
from tensorflow.keras import models
from PIL import Image
import numpy as np


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/canvas')
def canvas():
    return render_template('canvas.html')


@app.route('/import')
def import_picture():
    return render_template('import.html')


@app.route('/upload', methods=['POST'])
def upload_picture():
    file = request.files['file']
    if file:
        print("Image is loaded")
        # Prepare the prediction
        resized_image = preprocess_image(file)

        # Predict the image
        predictions = predict_digit(resized_image)

        # Extract the predicted class and confidence
        predicted_class = int(np.argmax(predictions))  # Convert NumPy scalar to int
        confidence = float(np.max(predictions))  # Convert Numpy scalar to float

        return jsonify({"prediction": predicted_class,
                        "confidence": confidence,
                        "probabilities": predictions.tolist()
                        })
    else:
        return jsonify({"error": "No file uploaded"}), 400


def preprocess_image(file, target_size=(28, 28)):
    """Resize the image"""
    image = Image.open(file).convert('L')  # Convert to grayscale
    img = image.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    return img_array[np.newaxis, :, :]  # Add batch dimension


def predict_digit(img):
    """
    Predict the image
    The model used for the system will be updated.
    """
    # Load the model
    model = models.load_model('./tf_practice/best_model.h5')
    return model.predict(img)


if __name__ == '__main__':
    app.run(debug=True)
