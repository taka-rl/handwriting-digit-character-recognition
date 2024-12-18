from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from build_models import (train_and_save_model, build_dense_model1,
                          build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3)


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


def predict_image(model, img_path, true_label: int):
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

    except FileNotFoundError:
        print(f"Error: File not found at {img_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalization uint8 -> float32
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255

    # For CNN models
    x_train_cnn, x_test_cnn = x_train.reshape(x_train.shape + (1,)), x_test.reshape(x_test.shape + (1,))

    # Load the model
    # -----------------------------------------------------------------------------
    # dense model1
    model_lr, history_lr = train_and_save_model(
        build_dense_model1,
        'digit_recognizer_dense1.h5',
        (x_train, y_train),
        (x_test, y_test)
    )

    # -----------------------------------------------------------------------------
    # dense model2
    model_mlp, history_mlp = train_and_save_model(
        build_dense_model2,
        'digit_recognizer_dense2.h5',
        (x_train, y_train),
        (x_test, y_test)
    )

    # -----------------------------------------------------------------------------
    # CNN model1
    model_cnn1, history_cnn1 = train_and_save_model(
        build_cnn_model1,
        'digit_recognizer_cnn1.h5',
        (x_train_cnn, y_train),
        (x_test_cnn, y_test)
    )

    # -----------------------------------------------------------------------------
    # CNN model2
    model_cnn2, history_cnn2 = train_and_save_model(
        build_cnn_model2,
        'digit_recognizer_cnn2.h5',
        (x_train_cnn, y_train),
        (x_test_cnn, y_test)
    )

    # -----------------------------------------------------------------------------
    # CNN model3
    model_cnn3, history_cnn3 = train_and_save_model(
        build_cnn_model3,
        'digit_recognizer_cnn3.h5',
        (x_train_cnn, y_train),
        (x_test_cnn, y_test)
    )

    # -----------------------------------------------------------------------------
    # Prediction
    models_dict = {"Dense1": model_lr,
                   "Dense2": model_mlp,
                   "CNN1": model_cnn1,
                   "CNN2": model_cnn2,
                   "CNN3": model_cnn3}

    # Show the prediction result
    for name, model in models_dict.items():
        predictions = model.predict(x_test)
        i: int = 12  # y_test index
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plot_image(i, predictions, x_test, y_test)
        plt.subplot(1, 2, 2)
        plt.title(f"Prediction Distribution: {name}")
        plot_value_array(predictions[i], true_label=y_test[i])
        plt.tight_layout()
    plt.show()

    # Predict and display results for external images
    for i in range(3):
        for name, model in models_dict.items():
            predict_image(model, f'../tf_practice/digits/{i}.png', true_label=i)
    plt.show()


if __name__ == '__main__':
    main()
