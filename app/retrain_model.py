from PIL import Image
import base64
import io
import numpy as np
import json
from sklearn.model_selection import train_test_split
from utilities import validate_image, preprocess_image
from gss import fetch_data_from_sheets
from models import model_digit


def create_test_dataset(records):
    """
    Create dataset based on the collected data from Google Spreadsheet

    Returns:
        tuple[np.ndarray, np.ndarray]: created dataset including data and its label
    """
    images, labels = [None] * len(records), [None] * len(records)

    for i, row in enumerate(records):
        if row['User_Corrected_Label'] is not None and row['User_Corrected_Label'] != "":
            # Only use data with corrected labels
            try:
                image_data = validate_image(row['Data'])
                image_data = base64.b64decode(image_data)
                image = preprocess_data_from_sheet(image_data)
                images[i], labels[i] = image, int(row['User_Corrected_Label'])
            except Exception as e:
                print(f"Error processing image: {e}")

    return np.array(images, dtype='float32'), np.array(labels, dtype='uint8')


def preprocess_data_from_sheet(image_data):
    """
    Preprocess collected digit data from Google Spreadsheet

    Returns:
        Preprocessed image
    """
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = preprocess_image(image)
    return image


def split_data(sheet_name: str, test_size: float = 0.2):
    """Split the data into train and test"""
    records = fetch_data_from_sheets(sheet_name)
    x_data, y_data = create_test_dataset(records)

    # Split into 80% training, 20% test data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

    return (x_train, y_train), (x_test, y_test)


def save_training_history(model_path: str, history) -> None:
    with open(model_path + '.json', 'w') as f:
        json.dump(history.history, f)


def save_model_json(model_path: str, model) -> None:
    model_json = model.to_json()
    with open(f'{model_path}_model_structure.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'{model_path}_weight.h5')


def retrain_model(sheet_name: str, epochs: int = 10, batch_size: int = 128) -> None:
    train_data, test_data = split_data(sheet_name)

    if train_data is None:
        # End the function if there is no training data
        return

    train_data = (train_data[0].reshape(-1, 28, 28, 1), train_data[1])
    test_data = (test_data[0].reshape(-1, 28, 28, 1), test_data[1])

    model_path = "artifacts/digit_dummy_retrained"

    model_digit_old = model_digit
    model_digit_old.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_digit.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    history_digit = model_digit.fit(*train_data,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=test_data,
                                    verbose=False)

    # Evaluation
    loss_digit, acc_digit = model_digit.evaluate(*test_data)
    print(f"Trained model: Accuracy: {acc_digit:.4f}, Loss: {loss_digit:.4f}")
    loss_digit_old, acc_digit_old = model_digit.evaluate(*test_data)
    print(f"Trained model: Accuracy: {acc_digit_old:.4f}, Loss: {loss_digit_old:.4f}")

    # Save the model and history
    model_digit.save(f"{model_path}.h5")
    save_model_json(model_path, model_digit)
    save_training_history(model_path, history_digit)
    print("Retraining Complete!")


if __name__ == '__main__':
    retrain_model("Digit_dummy")
