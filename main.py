from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
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
        image = Image.open(file).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to model's input size
        image_array = np.array(image) / 255.0  # Normalize pixel value

        # Add prediction model

        return jsonify({"prediction": "dummy_prediction"})
    else:
        return jsonify({"error": "No file uploaded"}), 400


if __name__ == '__main__':
    app.run(debug=True)
