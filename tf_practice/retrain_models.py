from app.gss import fetch_data_from_sheets
from tf_practice.src.recognition_system import RecognitionSystem
from tf_practice.src.model import Model
from tf_practice.src.utilities import plot_training_history
from sklearn.model_selection import train_test_split


def split_data(sheet_name, test_size=0.2):
    records = fetch_data_from_sheets(sheet_name)
    x_data, y_data = RecognitionSystem().create_test_dataset(records)

    # Split into 80% training, 20% test data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

    return (x_train, y_train), (x_test, y_test)


def main():
    # Initialization
    recognition_system = RecognitionSystem()
    model = Model(dataset_name="mnist", model_type="CNN", epochs=10)

    # Obtain dataset from Google Spreadsheet
    sheet_name = "Digit"
    model.train_data, model.test_data = split_data(sheet_name, test_size=0.2)

    # Preprocessed the dataset
    model.train_data = recognition_system.reshape_for_cnn(model.train_data)
    model.test_data = recognition_system.reshape_for_cnn(model.test_data)

    # Load the model
    if sheet_name == "Digit":
        model_path = "../tf_practice/digits/models/"
        model_name = "digit_recognizer_cnn3.h5"
    elif sheet_name == "Character":
        model_path = "../tf_practice/characters/models/byclass/"
        model_name = "character_recognizer_cnn3.h5"
    else:
        raise ValueError('sheet_name must be either Digit or Character.')

    model.load_model(model_path + model_name)

    # Retrain the model
    model.compile_model()
    history = model.train_model()

    # Prepare old model to compare
    model_old = Model(dataset_name="mnist", model_type="CNN", epochs=10)
    model_old.test_data = model.test_data
    model_old.load_model(model_path + model_name)
    model_old.compile_model()

    # Evaluation
    loss, acc = model.evaluate_model()
    loss_old, acc_old = model_old.evaluate_model()
    print(f"Trained model: Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    print(f"Old model: Accuracy: {acc_old:.4f}, Loss: {loss_old:.4f}")
    # label = ['CNN']
    # plot_training_history(history, label)

    # Save the model
    # model.save_model(model_path + model_name + "_retrained"), model.save_model_json()
    # model.save_training_history(model_path + model_name + "_retrained", history)


if __name__ == '__main__':
    main()
