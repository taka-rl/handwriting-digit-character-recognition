import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_image(idx: int, pred) -> None:
    prediction, true_label, img = np.argmax(pred[idx]), y_test[idx], x_test[idx]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[idx], cmap='gray')

    color = 'blue' if prediction == true_label else 'red'  # blue if correct, otherwise, red

    plt.xlabel(f'pred:{prediction}, {100*np.max(pred[idx]):.2f}% (true:{true_label}', color=color)


def plot_value_array(idx: int, pred) -> None:
    predictions_array, true_label = pred[idx], y_test[idx]
    plt.grid(False)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.yticks([])
    plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    plot[predicted_label].set_color('red')
    plot[true_label].set_color('blue')


def plot_training_history(histories, labels):
    plt.figure(figsize=(12, 5))
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_accuracy'], label=f'{label} Val Accuracy')
        plt.plot(history.history['accuracy'], '--', label=f'{label} Train Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.title("Model Training Accuracy Comparison")
    plt.show()


def build_dense_model1():
    """Build a simple Dense Neural Network model"""
    model = tf.keras.models.Sequential([
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def build_dense_model2():
    """Build a simple Dense Neural Network model"""
    model = tf.keras.models.Sequential([
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(64, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def build_cnn_model1():
    """
    Build a simple Convolutional Neural Network model

    The model is based on the following article.
    https://medium.com/@AMustafa4983/handwritten-digit-recognition-a-beginners-guide-638e0995c826
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
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    return model


def build_cnn_model2():
    """
    Build a simple Convolutional Neural Network model
    The model is based on the following article.
    https://medium.com/artificialis/get-started-with-computer-vision-by-building-a-digit-recognition-model-with-tensorflow-b2216823b90a

    The model follows the TinyVGG architecture,
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
        layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    return model


def train_and_save_model(model_builder, model_path, train_data, val_data, epochs=10, batch_size=128):
    if os.path.isfile(model_path):
        print(f"------ Loading existing model: {model_path} ------")
        model = models.load_model(model_path)
    else:
        model = model_builder()
        model.summary()
        print(f"Training new model: {model_path}")
        history = model.fit(*train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, verbose=False)
        model.save(model_path)
        return model, history
    return model, None


if __name__ == '__main__':
    '''
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
    x_train, x_test =  x_train.astype('float32') / 255, x_test.astype('float32') / 255

    # For CNN models
    x_train_cnn, x_test_cnn = x_train.reshape(x_train.shape + (1,)), x_test.reshape(x_test.shape + (1,))

    # -----------------------------------------------------------------------------
    # dense model1
    model_lr, history_lr = train_and_save_model(
        build_dense_model1,
        'models/digit_recognizer_dense1.h5',
        (x_train, y_train),
        (x_test, y_test)
    )
    # Evaluation
    dense_test_loss1, dense_test_acc1 = model_lr.evaluate(x_test, y_test)

    # -----------------------------------------------------------------------------
    # dense model2
    model_mlp, history_mlp = train_and_save_model(
        build_dense_model2,
        'models/digit_recognizer_dense2.h5',
        (x_train, y_train),
        (x_test, y_test)
    )
    # Evaluation
    dense_test_loss2, dense_test_acc2 = model_mlp.evaluate(x_test, y_test)

    # -----------------------------------------------------------------------------
    # CNN model1
    model_cnn1, history_cnn1 = train_and_save_model(
        build_cnn_model1,
        'models/digit_recognizer_cnn1.h5',
        (x_train, y_train),
        (x_test, y_test)
    )
    # Evaluation
    cnn_test_loss1, cnn_test_acc1 = model_cnn1.evaluate(x_test_cnn, y_test)

    # -----------------------------------------------------------------------------
    # CNN model2
    model_cnn2, history_cnn2 = train_and_save_model(
        build_cnn_model2,
        'models/digit_recognizer_cnn2.h5',
        (x_train, y_train),
        (x_test, y_test)
    )
    # Evaluation
    cnn_test_loss2, cnn_test_acc2 = model_cnn2.evaluate(x_test_cnn, y_test)

    # -----------------------------------------------------------------------------
    # Compare performance
    print(f"Dense Model Test1 Accuracy: {dense_test_acc1:.4f}")
    print(f"Dense Model Test2 Accuracy: {dense_test_acc2:.4f}")
    print(f"CNN Model Test1 Accuracy: {cnn_test_acc1:.4f}")
    print(f"CNN Model Test2 Accuracy: {cnn_test_acc2:.4f}")

    if not (history_lr is None, history_mlp is None, history_cnn1 is None, history_cnn2 is None):
        plot_training_history([history_lr, history_mlp, history_cnn1, history_cnn2],
                              ["Dense1", "Dense2", "CNN1", "CNN2"])

    '''
    plt.figure(1)
    plt.plot(history_lr.history['loss'], label='train')
    plt.plot(history_lr.history['val_loss'], label='val')
    plt.ylabel('loss')
    plt.legend()

    plt.figure(2)
    plt.plot(history_lr.history['accuracy'], label='train')
    plt.plot(history_lr.history['val_accuracy'], label='val')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # evaluation
    model_lr.evaluate(x_test, y_test)

    # Display the prediction result
    predictions = model_lr.predict(x_test)
    i: int = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i)
    plt.subplot(1, 2, 2)
    plot_value_array(i)
    plt.show()
    '''
