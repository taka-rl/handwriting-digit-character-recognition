from tf_practice.src.recognition_system import RecognitionSystem
from tf_practice.src.model import Model
from emnist import list_datasets
import os
from tf_practice.src.build_models import build_dense_model1, build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3
from tf_practice.src.utilities import plot_training_history


def main():
    print(list_datasets())

    # Initialization
    model_dnn1 = Model(dataset_name="emnist/byclass", model_builder=build_dense_model1, model_type="DNN", epochs=10)
    model_dnn2 = Model(dataset_name="emnist/byclass", model_builder=build_dense_model2, model_type="DNN", epochs=10)
    model_cnn1 = Model(dataset_name="emnist/byclass", model_builder=build_cnn_model1, model_type="CNN", epochs=10)
    model_cnn2 = Model(dataset_name="emnist/byclass", model_builder=build_cnn_model2, model_type="CNN", epochs=10)
    model_cnn3 = Model(dataset_name="emnist/byclass", model_builder=build_cnn_model3, model_type="CNN", epochs=10)
    recognition_system = RecognitionSystem()

    # Load and prepare the dataset
    model_dnn1.load_dataset()
    model_dnn1.train_data = recognition_system.preprocess_emnist_byclass(model_dnn1.train_data)
    model_dnn1.test_data = recognition_system.preprocess_emnist_byclass(model_dnn1.test_data)

    model_cnn2.train_data = recognition_system.reshape_for_cnn(model_dnn1.train_data)
    model_cnn2.test_data = recognition_system.reshape_for_cnn(model_dnn1.test_data)

    # Use shared preprocessed data
    model_dnn2.train_data, model_dnn2.test_data = model_dnn1.train_data, model_dnn1.test_data
    model_cnn2.train_data, model_cnn2.test_data = model_cnn1.train_data, model_cnn1.test_data
    model_cnn3.train_data, model_cnn3.test_data = model_cnn1.train_data, model_cnn1.test_data

    # Build and train the model
    # Paths and model names
    path = 'characters/models/byclass/'
    model_info = [
        ("character_recognizer_dnn1.h5", model_dnn1),
        ("character_recognizer_dnn2.h5", model_dnn2),
        ("character_recognizer_cnn1.h5", model_cnn1),
        ("character_recognizer_cnn2.h5", model_cnn2),
        ("character_recognizer_cnn3.h5", model_cnn3),
    ]

    histories, loss_list, acc_list = [None] * len(model_info), [None] * len(model_info), [None] * len(model_info)

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
            model.save_training_history(model_path, histories[i])
            model.save_model_json()

            # Evaluation
            loss, acc = model.evaluate_model()
            loss_list[i], acc_list[i] = loss, acc
            print(f"Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    '''
    # Create the training result plot
    json_files = [path+"character_recognizer_dnn1.h5.json",
                  path+"character_recognizer_dnn2.h5.json",
                  path+"character_recognizer_cnn1.h5.json",
                  path+"character_recognizer_cnn2.h5.json",
                  path+"character_recognizer_cnn3.h5.json"]
    
    plot_training_history_from_json(json_files, labels)
    '''

    # Plot
    # Filter out None values and pair them with their labels
    labels = ["DNN1", "DNN2", "CNN2"]
    histories_and_labels = [(history, label) for history, label in zip(histories, labels) if history is not None]

    # Unzip the filtered pairs back into two separate lists
    if histories_and_labels:
        valid_histories, valid_labels = zip(*histories_and_labels)
        plot_training_history(list(valid_histories), list(valid_labels))


if __name__ == "__main__":
    main()


