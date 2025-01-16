from tensorflow.keras import layers, models
import tensorflow as tf
import os
import json


def build_dense_model1(num_classes: int):
    """
    Build a Dense Neural Network model

    The model consists of the following layers.
    Input -> Flatten -> Dense

    Here’s a breakdown of the architecture:
        Flatten Layer:
            The Flatten layer converts the 2D image (28x28) into a 1D vector (784 elements).
            Dense layers require 1D input.
        Dense Layer:
            The Dense layer has 10 neurons to represent each digit class (0-9).
            Softmax is often used as the activation for the last layer of a classification network
            as it can produce a probability distribution as the result.
    """
    model = tf.keras.models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def build_dense_model2(num_classes: int):
    """
    Build a Dense Neural Network model

    The model consists of the following layers.
    Input -> Flatten -> Dense -> Dense -> Dense

    Here’s a breakdown of the architecture:
        Flatten Layer:
            The Flatten layer converts the 2D image (28x28) into a 1D vector (784 elements).
            Dense layers require 1D input.

        Two Hidden Layers (Dense Layers):
            They are fully connected layers with 64 neurons.
            elu, which stands for Exponential Liner Unit and represents the following equations.
                f(x) = x if x > 0, alpha * (exp(x) - 1) if x < 0

        Dense Layer:
            The Dense layer has 10 neurons to represent each digit class (0-9).
            Softmax is often used as the activation for the last layer of a classification network
            as it can produce a probability distribution as the result.
    """
    model = tf.keras.models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(64, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def build_cnn_model1(num_classes: int):
    """
    Build a Convolutional Neural Network model

    The model is based on the following article.
    https://medium.com/@AMustafa4983/handwritten-digit-recognition-a-beginners-guide-638e0995c826

    The model consists of the following layers.
    Conv2D -> Conv2D -> MaxPool2D -> Dropout -> Conv2D -> Conv2D -> MaxPool2D ->
        Dropout -> Flatten -> Dense -> Dropout -> Dense

    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def build_cnn_model2(num_classes: int):
    """
    Build a Convolutional Neural Network model
    The model is based on the following article.
    https://medium.com/artificialis/get-started-with-computer-vision-by-building-a-digit-recognition-model-with-tensorflow-b2216823b90a

    The model follows the TinyVGG architecture.
    Conv2D -> Conv2D -> MaxPool2D -> Conv2D -> Conv2D -> MaxPool2D -> Flatten -> Dense

    """
    model = tf.keras.Sequential([
        layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(28,  28,  1)),
        layers.Conv2D(10,  3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(10,  3, activation="relu"),
        layers.Conv2D(10,  3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def build_cnn_model3(num_classes: int):
    """
    Build a Convolutional Neural Network model
    The model is based on the following GitHub repository.
    https://github.com/maneprajakta/Digit_Recognition_Web_App

    The model consists of the following layers.
    Conv2D -> MaxPool2D -> Conv2D -> BatchNormalization -> Conv2D -> BatchNormalization -> Dropout ->
        Conv2D -> BatchNormalization -> Conv2D -> BatchNormalization -> Conv2D -> BatchNormalization ->
            Dropout -> Flatten -> Dropout -> Dense

    """
    model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_or_load_model(model_builder,
                        sys_name: str,
                        model_name: str,
                        train_data,
                        val_data,
                        epochs: int = 10,
                        batch_size: int = 128):
    """
    Train a model and save the model as .h5 the training result as JSON file, or load a model if the model exists.
    """

    model_path = os.path.join(f'{sys_name}/model', model_name)
    if os.path.isfile(model_path):
        print(f"------ Loading existing model: {model_path} ------")
        model = models.load_model(model_path, compile=False)
    else:
        if sys_name == 'digits':
            num_classes = 10
        elif sys_name == 'characters':
            num_classes = 26
        else:
            print("sys_name is an invalid value. You should choose 'digit' or 'character'.")
            raise ValueError

        model = model_builder(num_classes)
        model.summary()
        print("Compiling new model")
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(f"Training new model: {model_path}")
        history = model.fit(*train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, verbose=False)
        model.save(model_path)
        with open(model_path + '.json', 'w') as f:
            json.dump(history.history, f)
        return model, history
    return model, None
