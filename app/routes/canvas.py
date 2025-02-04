from flask import Blueprint, render_template, request, jsonify
from PIL import Image
import numpy as np
import base64
import io
from app.models import model_digit, model_character, character_list
from app.utilities import validate_image, preprocess_image, reshape_for_cnn
from app.gss import save_to_sheet


canvas_bp = Blueprint('canvas', __name__)


@canvas_bp.route('/canvas-digit')
def canvas_digit():
    return render_template('canvas_digit.html')


@canvas_bp.route('/canvas-character')
def canvas_character():
    return render_template('canvas_character.html')


@canvas_bp.route('/submit-digit', methods=['POST'])
def submit_digit_drawing():
    data = request.json.get('image', None)
    encoded = validate_image(data)
    try:
        # Decode the Base 64 image
        image_data = base64.b64decode(encoded)

        # Preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('L')
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
                        "probabilities": predictions.tolist(),
                        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@canvas_bp.route('/submit-character', methods=['POST'])
def submit_character_drawing():
    data = request.json.get('image', None)
    encoded = validate_image(data)
    try:
        # Decode the Base 64 image
        image_data = base64.b64decode(encoded)

        # Preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('L')
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
                        "lower_probabilities": lower_predictions.tolist(),
                        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@canvas_bp.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """
    Submit a feedback from users to Google Spreadsheet

    """
    data = request.json
    predict_label = data.get('predictedLabel')
    correct_label = data.get('correct_label')
    image = data.get('image')
    confidence = data.get('confidence')
    confidence = confidence.replace('Confidence: ', '')

    try:
        sheet_name = 'Digit' if correct_label.isdigit() else 'Character'

        # Save the input drawn digit data and prediction data to Google Spreadsheet
        save_to_sheet(sheet_name, str(image), str(predict_label), confidence, str(correct_label))

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
