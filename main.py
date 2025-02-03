from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import base64
import io
import os
from gss import save_to_sheet


with open("static/models/model_digit.json", "r") as json_file:
    model_digit_json = json_file.read()

with open("static/models/model_character.json", "r") as json_file:
    model_character_json = json_file.read()

model_digit, model_character = model_from_json(model_digit_json), model_from_json(model_character_json)
model_digit.load_weights("static/models/model_digit_weights.h5")
model_character.load_weights("static/models/model_character_weights.h5")

character_list = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')

del model_digit_json, model_character_json

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/canvas-digit')
def canvas_digit():
    return render_template('canvas_digit.html')


@app.route('/canvas-character')
def canvas_character():
    return render_template('canvas_character.html')


@app.route('/submit-digit', methods=['POST'])
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


@app.route('/submit-character', methods=['POST'])
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


@app.route('/submit-feedback', methods=['POST'])
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


@app.route('/import-digit')
def import_digit_picture():
    return render_template('import_digit.html')


@app.route('/import-character')
def import_character_picture():
    return render_template('import_character.html')


@app.route('/upload-digit', methods=['POST'])
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


@app.route('/upload-character', methods=['POST'])
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


def reshape_for_cnn(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape + (1,))


def preprocess_image(image, target_size=(28, 28)):
    """Resize and normalize the image."""
    img = image.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    return img_array[np.newaxis, :, :]  # Add batch dimension


def validate_image(image):
    """
    Validate and extract the image data.

    Parameter:
        data: the input drawn image, which is Base 64 image
    Returns:
        encoded: extracted only image data from the image
    """
    if image is None:
        return jsonify({"error": "No image data provided"})
    if ',' in image:
        header, encoded = image.split(',', 1)
        return encoded
    else:
        return jsonify({'error': 'Invalid Base64 image format'}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
