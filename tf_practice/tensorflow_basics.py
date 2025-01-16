import tensorflow as tf
import matplotlib.pyplot as plt
from build_models import (train_or_load_model, build_dense_model1,
                          build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3)


def plot_training_history(histories, labels):
    plt.figure(figsize=(12, 5))
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_accuracy'], label=f'{label} Val Accuracy')
        plt.plot(history.history['accuracy'], '--', label=f'{label} Train Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.title("Model Training Accuracy Comparison")
    plt.show()


if __name__ == '__main__':
    '''
    This script is used with build_models.py for learning the basics of TensorFlow, 
    including the following basic procedure.
    The basic procedure of TensorFlow is as follows:
    1: Prepare a dataset(MNIST for digit)
        Load the dataset
        Normalize the dataset
        
    * If you load a model, load the model instead of 2 and 3 steps, which the functions are defined in build_models.py.
    * Load a model 
        model = models.load_model(model_path)
        
    2: Build a model
        Choose a model type such as fully connected and CNN
        Create a model by using the following code
            model = tf.keras.Sequential([
                layers.Input, layers.Flatten, layers.Conv2D and so on
    3: Train the model
        history = model.fit()
    4: Save the model and the training result as a JSON file
        model.save() for the model
        with open(model_path + '.json', 'w') as f: # for the training result 
            json.dump(history.history, f)
    5: Plot the training result with Matplotlib and Evaluation the model   
    
    About the MNIST dataset: 
        It contains two sets such as a dataset including 60000 28x28 grayscale images of the 10 digits(0-9)
        and a test set including 10000 images.
        
    It returns the Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
        x_train: 
            This is the training data, consisting of uint8 NumPy array of grayscale image data 
            with shapes of (60000, 28, 28). Pixel values range from 0 to 255.
        y_train: 
            This is the training data, consisting of unit8 NumPy array of digit labels (integer in range 0 to 9) 
            with shape (60000,).
        x_test: 
            This is the test data, consisting of uint8 NumPy array of grayscale image data 
            with shapes of (10000, 28, 28). Pixel values range from 0 to 255.
        y_test: 
            This is the test data, consisting of unit8 NumPy array of digit labels (integer in range 0 to 9) 
            with shape (10000,).
    
    For Dense Neural Networks (DNN), we do not need to reshape the input images into (28, 28, 1).
    Instead, we need to flatten the image directly into a 1D vector, 
    because Dense (fully connected) layers require 1D input.
    
    Since the Conv2D layer in a convolutional model requires the 3D input (height, width, color_channels),
    it is necessary to reshape the training and test data to have the missing color_channels dimension, 
    which is 1 for grayscale images. 
    x_train = x_train.reshape(x_train.shape + (1,)
    x_train.shape -> (60000, 28, 28, 1)
    
    About training a model
    y_onehot_train = tf.one_hot(y_train, 10)  # if you use loss='categorical_crossentropy
    validation_data=[x_test, y_test] or validation_split=0.2
    
    
    '''
    # Load MNIST for digit recognition
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalization uint8 -> float32
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255

    # For CNN models
    x_train_cnn, x_test_cnn = x_train.reshape(x_train.shape + (1,)), x_test.reshape(x_test.shape + (1,))

    # -----------------------------------------------------------------------------
    # dense model1
    model_lr, history_lr = train_or_load_model(build_dense_model1, 'digits', 'digit_recognizer_dense1.h5',
                                               (x_train, y_train), (x_test, y_test))
    # Evaluation
    dense_test_loss1, dense_test_acc1 = model_lr.evaluate(x_test, y_test)

    # -----------------------------------------------------------------------------
    # dense model2
    model_mlp, history_mlp = train_or_load_model(build_dense_model2, 'digits', 'digit_recognizer_dense2.h5',
                                                 (x_train, y_train), (x_test, y_test))
    # Evaluation
    dense_test_loss2, dense_test_acc2 = model_mlp.evaluate(x_test, y_test)

    # -----------------------------------------------------------------------------
    # CNN model1
    model_cnn1, history_cnn1 = train_or_load_model(build_cnn_model1, 'digits', 'digit_recognizer_cnn1.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # Evaluation
    cnn_test_loss1, cnn_test_acc1 = model_cnn1.evaluate(x_test_cnn, y_test)

    # -----------------------------------------------------------------------------
    # CNN model2
    model_cnn2, history_cnn2 = train_or_load_model(build_cnn_model2, 'digits', 'digit_recognizer_cnn2.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # Evaluation
    cnn_test_loss2, cnn_test_acc2 = model_cnn2.evaluate(x_test_cnn, y_test)

    # -----------------------------------------------------------------------------
    # CNN model3
    model_cnn3, history_cnn3 = train_or_load_model(build_cnn_model3, 'digits', 'digit_recognizer_cnn3.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # Evaluation
    cnn_test_loss3, cnn_test_acc3 = model_cnn3.evaluate(x_test_cnn, y_test)

    # -----------------------------------------------------------------------------
    # Compare performance
    print(f"Dense Model Test1 Accuracy: {dense_test_acc1:.4f}")
    print(f"Dense Model Test2 Accuracy: {dense_test_acc2:.4f}")
    print(f"CNN Model Test1 Accuracy: {cnn_test_acc1:.4f}")
    print(f"CNN Model Test2 Accuracy: {cnn_test_acc2:.4f}")
    print(f"CNN Model Test3 Accuracy: {cnn_test_acc3:.4f}")

    # Plot
    # Filter out None values and pair them with their labels
    histories_and_labels = [(history, label) for history, label in zip(
        [history_lr, history_mlp, history_cnn1, history_cnn2, history_cnn3],
        ["Dense1", "Dense2", "CNN1", "CNN2", "CNN3"]
    ) if history is not None]

    # Unzip the filtered pairs back into two separate lists
    if histories_and_labels:
        valid_histories, valid_labels = zip(*histories_and_labels)
        plot_training_history(list(valid_histories), list(valid_labels))
