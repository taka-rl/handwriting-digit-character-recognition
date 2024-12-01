import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def plot_image(idx: int) -> None:
    prediction, true_label, img = np.argmax(predictions[idx]), y_test[idx], x_test[idx]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[idx], cmap='gray')

    color = 'blue' if prediction == true_label else 'red'  # blue if correct, otherwise, red

    plt.xlabel(f'pred:{prediction}, {100*np.max(predictions[idx]):.2f}% (true:{true_label}', color=color)


def plot_value_array(idx: int) -> None:
    predictions_array, true_label = predictions[idx], y_test[idx]
    plt.grid(False)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.yticks([])
    plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    plot[predicted_label].set_color('red')
    plot[true_label].set_color('blue')


if __name__ == '__main__':

    # Load MNIST for digit recognition
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalization
    x_train, x_test = x_train / 255, x_test / 255

    # Build a model
    model_lr = tf.keras.models.Sequential([
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model_lr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Show the summary of the model
    model_lr.summary()

    model_mlp = tf.keras.models.Sequential([
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(64, activation='elu'),
        layers.Dense(64, activation='elu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Show the summary of the model
    model_mlp.summary()

    # Train the model
    # y_onehot_train = tf.one_hot(y_train, 10)  # if you use loss='categorical_crossentropy
    # validation_data=[x_test, y_test] or validation_split=0.2
    history_lr = model_lr.fit(x_train,
                              y_train,
                              epochs=10,
                              batch_size=128,
                              validation_data=[x_test, y_test],
                              verbose=False)

    print(history_lr.history)

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
