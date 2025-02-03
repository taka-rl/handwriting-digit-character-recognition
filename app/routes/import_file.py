from flask import Blueprint, render_template, request, jsonify
from PIL import Image
import numpy as np
from app.models import model_digit, model_character, character_list
from app.utilities import preprocess_image, reshape_for_cnn


import_file_bp = Blueprint('import_file', __name__)


@import_file_bp.route('/import-digit')
def import_digit_picture():
    return render_template('import_digit.html')


@import_file_bp.route('/import-character')
def import_character_picture():
    return render_template('import_character.html')


@import_file_bp.route('/upload-digit', methods=['POST'])
def upload_digit_picture():
    file = request.files['file']
    if file:
        # Prepare the prediction
        image = Image.open(file).convert('L')  # Convert to grayscale
        processed_image = preprocess_image(image)

        # Reshape for CNN
        processed_image = reshape_for_cnn(processed_image)

        # Predict the image
        predictions = model_digit.predict(processed_image)

        # Extract the predicted class and confidence
        predicted_class = int(np.argmax(predictions))  # Convert NumPy scalar to int
        confidence = float(np.max(predictions))  # Convert Numpy scalar to float

        return jsonify({"prediction": predicted_class,
                        "confidence": confidence,
                        "probabilities": predictions.tolist()
                        })
    else:
        return jsonify({"error": "No file uploaded"}), 400


@import_file_bp.route('/upload-character', methods=['POST'])
def upload_character_picture():
    file = request.files['file']
    if file:
        # Prepare the prediction
        image = Image.open(file).convert('L')  # Convert to grayscale
        processed_image = preprocess_image(image)

        # Reshape for CNN
        processed_image = reshape_for_cnn(processed_image)

        # Predict the image
        predictions = model_character.predict(processed_image)

        # Extract the predicted class and confidence
        predicted_class = int(np.argmax(predictions))  # Convert NumPy scalar to int
        confidence = float(np.max(predictions))  # Convert Numpy scalar to float

        # Split predictions into uppercase and lowercase
        upper_predictions = predictions[0][:26]
        lower_predictions = predictions[0][26:]

        return jsonify({"prediction": character_list[predicted_class],
                        "confidence": confidence,
                        "upper_probabilities": upper_predictions.tolist(),
                        "lower_probabilities": lower_predictions.tolist()
                        })
    else:
        return jsonify({"error": "No file uploaded"}), 400

