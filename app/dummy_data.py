from PIL import Image
import numpy as np
import base64
import io
import time
from tensorflow.keras.datasets import mnist
from app.gss import save_to_sheet, fetch_data_from_sheets
from app.retrain_model import create_test_dataset
from app.models import model_digit
from tf_practice.src.utilities import create_prediction_plot


def generate_dummy_image():
    """
    Generate dummy image data.

    Returns:
        generated base64 dummy data.
    """
    img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    image_pil = Image.fromarray(img)
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")

    return 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode()


def insert_dummy_data(sheet_name: str = "Digit_dummy", data_length: int = 10) -> None:
    """
    Insert generated dummy data in the sheet.

    Parameters:
        sheet_name: a sheet name in the targeted Google Spreadsheet
        data_length: the number of dummy data to insert in the sheet

    """
    print("Inserting dummy data into Google Sheets...")

    for i in range(data_length):
        image_data = generate_dummy_image()
        predicted_label = np.random.randint(0, 10)  # Random digit (0-9)
        confidence = round(np.random.uniform(0.6, 1.0), 2)  # Confidence (60%-100%)
        correct_label = predicted_label
        save_to_sheet(sheet_name, image_data, predicted_label, confidence, correct_label)
        time.sleep(1)

    print("Dummy data inserted!")


def generate_rotate_mnist_data(rotation_range: int = 30):
    """
    Select a random digit from MNIST dataset, apply random rotation, and convert to base64.

    Parameters:
        rotation_range (int): Maximum degrees to rotate the image.

    Returns:
        base64 encoded image data and its label.
    """
    # Load MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()

    # Select a random MNIST digit
    idx = np.random.randint(0, x_train.shape[0])  # Random index
    img = x_train[idx]  # Get image from MNIST dataset (already 28x28)

    # Convert to PIL image
    image_pil = Image.fromarray(img)

    # Apply random rotation within the specified range
    angle = np.random.uniform(-rotation_range, rotation_range)
    rotated_image = image_pil.rotate(angle, resample=Image.BILINEAR)

    # Convert to base64
    buffered = io.BytesIO()
    rotated_image.save(buffered, format="PNG")
    return 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode(), y_train[idx]


def insert_generated_mnist_data(sheet_name: str, data_length: int = 10) -> None:
    """
    Insert generated MNIST data in the sheet

    Parameters:
        sheet_name: a sheet name in the targeted Google Spreadsheet
        data_length: the number of dummy data to insert in the sheet

    """
    print("Inserting generated MNIST data into Google Sheets...")

    for i in range(data_length):
        image_data, correct_label = generate_rotate_mnist_data()
        save_to_sheet(sheet_name, image_data, None, None, str(correct_label))
        time.sleep(1)

    print("Generated MINST data inserted!")


def check_dummy_image(sheet_name: str):
    # Collect data from the spreadsheet
    records = fetch_data_from_sheets(sheet_name)

    # Create and reshape the dataset
    x_data, y_data = create_test_dataset(records)
    x_data_cnn = x_data.reshape(-1, 28, 28, 1)
    x_data = x_data.reshape(-1, 28, 28)

    # Choose a random index
    idx = np.random.randint(0, x_data.shape[0])

    # Load a model from the digit recognition
    model = model_digit

    # Predict the image
    predictions = model.predict(x_data_cnn)

    # Extract the predicted class and confidence
    predicted_class = int(np.argmax(predictions[idx]))  # Convert NumPy scalar to int
    confidence = float(np.max(predictions[idx]))  # Convert Numpy scalar to float

    # Create the plot
    create_prediction_plot("Digit_dummy", predictions[0], predicted_class, confidence, y_data[idx], x_data_cnn[idx])


if __name__ == '__main__':
    insert_generated_mnist_data("Digit_dummy")
    # check_dummy_image("Digit_dummy")

