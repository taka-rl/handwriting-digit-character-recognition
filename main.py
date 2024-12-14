from flask import Flask, request, jsonify, render_template
from tensorflow.keras import models
from PIL import Image
import numpy as np
import base64
import io


# Load the model globally
model = models.load_model('./tf_practice/best_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/canvas')
def canvas():
    return render_template('canvas.html')


@app.route('/submit', methods=['POST'])
def submit_drawing():
    data = request.json.get('image', None)
    if data is None:
        return jsonify({"error": "No image data provided"})
    try:
        if ',' in data:
            header, encoded = data.split(',', 1)
        else:
            return jsonify({'error': 'Invalid Base64 image format'}), 400

        # Decode the Base 64 image
        image_data = base64.b64decode(encoded)

        # Preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('L')
        processed_image = preprocess_image(image)

        # Predict the image
        predictions = predict_digit(processed_image)

        # Extract the predicted class and confidence
        predicted_class = int(np.argmax(predictions))  # Convert NumPy scalar to int
        confidence = float(np.max(predictions))  # Convert Numpy scalar to float

        return jsonify({"prediction": predicted_class,
                        "confidence": confidence,
                        "probabilities": predictions.tolist()
                        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/import')
def import_picture():
    return render_template('import.html')


@app.route('/upload', methods=['POST'])
def upload_picture():
    file = request.files['file']
    if file:
        print("Image is loaded")
        # Prepare the prediction
        image = Image.open(file).convert('L')  # Convert to grayscale
        processed_image = preprocess_image(image)

        # Predict the image
        predictions = predict_digit(processed_image)

        # Extract the predicted class and confidence
        predicted_class = int(np.argmax(predictions))  # Convert NumPy scalar to int
        confidence = float(np.max(predictions))  # Convert Numpy scalar to float

        return jsonify({"prediction": predicted_class,
                        "confidence": confidence,
                        "probabilities": predictions.tolist()
                        })
    else:
        return jsonify({"error": "No file uploaded"}), 400


def preprocess_image(image, target_size=(28, 28)):
    """Resize and normalize the image."""
    img = image.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    return img_array[np.newaxis, :, :]  # Add bathtch dimension


def predict_digit(img):
    """
    Predict the image
    The model used for the system will be updated.
    """
    return model.predict(img)


if __name__ == '__main__':
    app.run(debug=True)
