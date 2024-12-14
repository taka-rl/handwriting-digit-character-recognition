from PIL import Image
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_image(img, target_size=(28, 28)):
    img = img.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    return img_array[np.newaxis, :, :]  # Add batch dimension


def plot_image(idx: int, predictions, x_test, y_test) -> None:
    prediction, true_label, img = np.argmax(predictions[idx]), y_test[idx], x_test[idx]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[idx], cmap='gray')

    color = 'blue' if prediction == true_label else 'red'  # blue if correct, otherwise, red
    confidence = 100 * np.max(predictions[idx])

    plt.xlabel(f'Pred: {prediction}, {confidence:.2f}% (True: {true_label}', color=color)


def plot_value_array(predictions_array, true_label=None) -> None:
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)
    plot[predicted_label].set_color('red')
    if true_label is not None:
        plot[true_label].set_color('blue')


def predict_image(img_path, true_label: int):
    try:
        # Load and preprocess the image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = preprocess_image(img)

        # Prediction
        pred = model.predict(img_array)

        # Visualization
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img_array[0], cmap='gray')
        plt.title("Resized Image")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plot_value_array(pred[0], true_label)
        plt.title("Prediction Distribution")
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {img_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    # Load MNIST for digit recognition
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalization
    x_train, x_test = x_train / 255, x_test / 255

    # Build the model
    if os.path.isfile('../tf_practice/best_model.h5'):
        # Load the existing model
        print("------ Loading the existing model ------")
        model = models.load_model('../tf_practice/best_model.h5')

    else:
        # If not, build the model
        print("------ Building a model ------")
        input_shape = x_train.shape[1:]  # Dynamically determine input shape
        model = tf.keras.models.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Show the summary of the model
        model.summary()

        # Train the model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]

        model.fit(x_train,
                  y_train,
                  epochs=20,
                  batch_size=128,
                  validation_data=(x_test, y_test),  # validation_data=[x_test, y_test] or validation_split=0.2
                  callbacks=callbacks,
                  verbose=False)

    # Evaluation
    model.evaluate(x_test, y_test)

    # Show the prediction result
    predictions = model.predict(x_test)
    i: int = 12  # y_test index
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plot_image(i, predictions, x_test, y_test)
    plt.subplot(1, 2, 2)
    plt.title("Prediction Distribution")
    plot_value_array(predictions[i], true_label=y_test[i])
    plt.tight_layout()
    plt.show()

    # Predict and display results for external images
    for i in range(3):
        predict_image(f'../tf_practice/digits/{i}.png', true_label=i)
    plt.show()
