from recognition_system import RecognitionSystem
from model import Model
import os


def main():
    # Initialization
    model_path = '../tf_practice/digits/models/digit_recognizer_cnn4.h5'
    model = Model(dataset_name="emnist", model_type="CNN", epochs=5)
    recognition_system = RecognitionSystem()

    # Load a model
    if os.path.isfile(model_path):
        print(f"------ Loading existing model: {model_path} ------")
        model.load_model(model_path)
    else:
        # if the model doesn't exist
        print('Load and preprocess dataset')
        model.load_dataset()
        if model.dataset_name == 'mnist':
            model.train_data = recognition_system.preprocess_mnist(model.train_data)
            model.test_data = recognition_system.preprocess_mnist(model.test_data)
            if model.model_type == 'cnn':
                model.train_data = recognition_system.reshape_for_cnn(model.train_data)
                model.test_data = recognition_system.reshape_for_cnn(model.test_data)

        elif model.dataset_name == 'emnist':
            model.train_data = recognition_system.preprocess_emnist(model.train_data, cnn=(model.model_type == 'CNN'))
            model.test_data = recognition_system.preprocess_emnist(model.test_data, cnn=(model.model_type == 'CNN'))

        print('Build a model')
        model.build_model()
        print('Compile the model')
        model.compile_model()
        print('Train the model')
        model.train_model()
        print('Evaluate the model')
        test_loss, test_acc = model.evaluate_model()
        print(f"Test Loss: {test_loss:.2f}")
        print(f"Test Accuracy: {test_acc:.2f}")

    # Predict the imported image
    # Import the image
    image_path = '../tf_practice/digits/test_images/0.png'
    image = recognition_system.import_image(image_path)
    if model.model_type == 'CNN':
        image = recognition_system.reshape_for_cnn(image)
    # Predict
    predictions = model.predict_data(image)
    predicted_class, confidence, _, _ = recognition_system.get_prediction_info(predictions)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
