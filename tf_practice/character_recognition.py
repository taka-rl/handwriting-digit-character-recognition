from recognition_system import RecognitionSystem
from model import Model
from emnist import list_datasets
import os
from build_models import build_dense_model1, build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3
import matplotlib.pyplot as plt
import json


def plot_training_history_from_json(json_files, labels):
    """
    Plot training history from JSON files.

    Parameters:
        json_files (list): List of file paths to JSON files.
        labels (list): Corresponding labels for the models.
    """
    plt.figure(figsize=(12, 5))
    for json_file, label in zip(json_files, labels):
        with open(json_file, 'r') as f:
            history = json.load(f)

        # Plot training and validation accuracy
        plt.plot(history['val_accuracy'], label=f'{label} Val Accuracy')
        plt.plot(history['accuracy'], '--', label=f'{label} Train Accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.title("Model Training Accuracy Comparison")
    plt.show()


def main():
    print(list_datasets())

    # Initialization
    model_dnn1 = Model(dataset_name="emnist", model_builder=build_dense_model1, model_type="DNN", epochs=10)
    model_dnn2 = Model(dataset_name="emnist", model_builder=build_dense_model2, model_type="DNN", epochs=10)
    model_cnn1 = Model(dataset_name="emnist", model_builder=build_cnn_model1, model_type="CNN", epochs=10)
    model_cnn2 = Model(dataset_name="emnist", model_builder=build_cnn_model2, model_type="CNN", epochs=10)
    model_cnn3 = Model(dataset_name="emnist", model_builder=build_cnn_model3, model_type="CNN", epochs=10)
    recognition_system = RecognitionSystem()

    # Load and prepare the dataset
    model_dnn1.load_dataset()
    model_dnn1.train_data = recognition_system.preprocess_emnist(model_dnn1.train_data)
    model_dnn1.test_data = recognition_system.preprocess_emnist(model_dnn1.test_data)

    model_cnn1.load_dataset()
    model_cnn1.train_data = recognition_system.preprocess_emnist(model_cnn1.train_data)
    model_cnn1.train_data = recognition_system.reshape_for_cnn(model_cnn1.train_data)
    model_cnn1.test_data = recognition_system.preprocess_emnist(model_cnn1.test_data)
    model_cnn1.test_data = recognition_system.reshape_for_cnn(model_cnn1.test_data)

    # Use shared preprocessed data
    model_dnn2.train_data, model_dnn2.test_data = model_dnn1.train_data, model_dnn1.test_data
    model_cnn2.train_data, model_cnn2.test_data = model_cnn1.train_data, model_cnn1.test_data
    model_cnn3.train_data, model_cnn3.test_data = model_cnn1.train_data, model_cnn1.test_data

    # Build and train the model
    # Paths and model names
    path = '../tf_practice/characters/models/'
    model_info = [
        ("character_recognizer_dnn1.h5", model_dnn1),
        ("character_recognizer_dnn2.h5", model_dnn2),
        ("character_recognizer_cnn1.h5", model_cnn1),
        ("character_recognizer_cnn2.h5", model_cnn2),
        ("character_recognizer_cnn3.h5", model_cnn3),
    ]

    histories, loss_list, acc_list = [None] * len(model_info), [] * len(model_info), [] * len(model_info)

    for i, (name, model) in enumerate(model_info):
        model_path = os.path.join(path, name)

        if os.path.isfile(model_path):
            print(f"------ Loading existing model: {model_path} ------")
            model.load_model(model_path)
        else:
            print(f"------ Building and training model: {name} ------")
            model.build_model()
            model.compile_model()
            histories[i] = model.train_model()
            model.save_model(model_path)
            model.save_model_json(model_path, histories[i])

            # Evaluation
            loss, acc = model.evaluate_model()
            loss_list[i], acc_list[i] = loss, acc
            print(f"Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Create the training result plot
    json_files = [path+"character_recognizer_dnn1.h5.json",
                  path+"character_recognizer_dnn2.h5.json",
                  path+"character_recognizer_cnn1.h5.json",
                  path+"character_recognizer_cnn2.h5.json",
                  path+"character_recognizer_cnn3.h5.json"]
    labels = ["DNN1", "DNN2", "CNN1", "CNN2", "CNN3"]
    plot_training_history_from_json(json_files, labels)


if __name__ == "__main__":
    main()


