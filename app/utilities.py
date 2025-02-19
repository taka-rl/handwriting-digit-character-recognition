from flask import jsonify
import numpy as np


def reshape_for_cnn(data: np.ndarray) -> np.ndarray:
    """
    Reshape the data for CNN models.
    The input shape is expected (batch_size, 28, 28)

    Returns:
        reshaped data followed (batch_size, 28, 28, 1)
    """

    return data.reshape(data.shape + (1,))


def preprocess_image(image, target_size: tuple = (28, 28)):
    """
    Preprocess the input image: resize, normalize, and add batch dimension.

    Parameters:
        image (PIL.Image.Image): Input image to preprocess.
        target_size (tuple): Target size for resizing (width, height).

    Returns:
        np.ndarray: Preprocessed image ready for the model.
    """
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
