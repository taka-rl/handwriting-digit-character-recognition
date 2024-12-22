from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from build_models import (train_and_save_model, build_dense_model1,
                          build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3)


def preprocess_image(img: Image.Image, target_size=(28, 28)) -> np.ndarray:
    """
    Preprocess the input image: resize, normalize, and add batch dimension.

    Parameters:
        img (PIL.Image.Image): Input image to preprocess.
        target_size (tuple): Target size for resizing (width, height).

    Returns:
        np.ndarray: Preprocessed image ready for the model.
    """
    img = img.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    return img_array[np.newaxis, :, :]  # Add batch dimension


def reshape_for_cnn(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape + (1,))


@tf.function
def predict_image(model, x_test):
    """
    Get predictions for the input data (decorated with tf.function to prevent retracing).
    """
    return model(x_test, training=False)


def predict_and_plot(model, model_name, data, labels, idx):
    """Predict and plot results for a given model."""
    create_prediction_plot(model, model_name, x_test=data, y_test=labels, idx=idx)


def create_prediction_plot(model, model_name, x_test, y_test, idx) -> None:
    """
    Create a prediction plot including the specific input image placed on the left side of the plot,
    the prediction result and the distribution bar chart placed on the right sice of the plot.

    Parameters:
        model: The trained TensorFlow/Keras model for predictions.
        model_name (str): The name of the model.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray or int): True labels corresponding to `x_test`.
        idx (int or None): Index of the test image. If `None`, the input is treated as a single image.
    """
    # Predict
    predictions = predict_image(model, x_test)

    # Get the result
    if idx is None:
        prediction, confidence, img = np.argmax(predictions[0]), 100 * np.max(predictions[0]), x_test[0]
        true_label = y_test
    else:
        prediction, confidence, img = np.argmax(predictions[idx]), 100 * np.max(predictions[idx]), x_test[idx]
        true_label = y_test[idx]

    # Set the plot size
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f'Input image and prediction result', fontsize=16)

    # Create a plot for the input image
    axes[0].set_title('Input Image', fontsize=10)
    axes[0].grid(False)
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')

    # Create a plot for the prediction result
    axes[1].set_title(f'{model_name}: Prediction Distribution\n'
                      f'Prediction: {prediction}, {confidence:.2f}% (True: {true_label})', fontsize=12)
    axes[1].grid(False)
    if idx is None:
        axes[1].bar(range(10), predictions[0], color="#777777")
    else:
        axes[1].bar(range(10), predictions[idx], color="#777777")
    axes[1].set_ylim([0, 1])

    # Highlight the bar chart with blue if the prediction is correct
    axes[1].patches[prediction].set_facecolor('red')
    if true_label == prediction:
        axes[1].patches[prediction].set_facecolor('blue')

    # Adjust the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalization uint8 -> float32
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255

    # For CNN models
    x_train_cnn, x_test_cnn = reshape_for_cnn(x_train), reshape_for_cnn(x_test)

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
    models_dense_dict = {"Dense1": model_lr, "Dense2": model_mlp}
    models_cnn_dict = {"CNN1": model_cnn1, "CNN2": model_cnn2, "CNN3": model_cnn3}
    models_dict = {"Dense1": model_lr, "Dense2": model_mlp, "CNN1": model_cnn1, "CNN2": model_cnn2, "CNN3": model_cnn3}

    # Plot the test image from MNIST and the prediction result (each)
    '''
    for model_name, model in models_dense_dict.items():
        predict_and_plot(model, model_name, x_test, y_test, idx=12)
    for model_name, model in models_cnn_dict.items():
        predict_and_plot(model, model_name, x_test_cnn, y_test, idx=12)
    '''
    # Plot the test image created by myself and the prediction result
    for i in range(3):
        for model_name, model in models_dict.items():
            try:
                img_path = f'../tf_practice/digits/{i}.png'
                true_label = i

                # Load and preprocess the image
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = preprocess_image(img)

                # Reshape the data if the model is CNN
                img_array = reshape_for_cnn(img_array) if 'CNN' in model_name else img_array

                # Predict and plot
                predict_and_plot(model, model_name, img_array, true_label, idx=None)

            except FileNotFoundError:
                print(f"Error: File not found at {img_path}")
            except Exception as e:
                print(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
