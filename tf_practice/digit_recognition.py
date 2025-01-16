from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_practice.src.build_models import (train_or_load_model, build_dense_model1,
                                          build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3)
from tf_practice.src.utilities import create_prediction_plot, create_combined_prediction_plot


def load_mnist():
    mnist = tf.keras.datasets.mnist
    return mnist.load_data()


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
def predict_image(model: tf.keras.models.Sequential, x_test: np.ndarray):
    """
    Get predictions for the input data (decorated with tf.function to prevent retracing).
    """
    return model(x_test, training=False)


def get_prediction_info(predictions: np.ndarray, idx: int = 0, x_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Extract prediction, confidence, true label, and input image for visualization.
    """
    prediction = np.argmax(predictions[idx])
    confidence = 100 * np.max(predictions[idx])
    if y_test is None:
        return prediction, confidence, None, None
    else:
        img, true_label = x_test[idx], y_test[idx]
        return prediction, confidence, img, true_label


def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Normalization uint8 -> float32
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255

    # For CNN models
    x_train_cnn, x_test_cnn = reshape_for_cnn(x_train), reshape_for_cnn(x_test)

    # Load the model
    # -----------------------------------------------------------------------------
    # dense model1
    model_lr, history_lr = train_or_load_model(build_dense_model1, 'digits', 'digit_recognizer_dense4.h5',
                                               (x_train, y_train), (x_test, y_test))
    # -----------------------------------------------------------------------------
    # dense model2
    model_mlp, history_mlp = train_or_load_model(build_dense_model2, 'digits', 'digit_recognizer_dense5.h5',
                                                 (x_train, y_train), (x_test, y_test))
    # -----------------------------------------------------------------------------
    # CNN model1
    model_cnn1, history_cnn1 = train_or_load_model(build_cnn_model1, 'digits', 'digit_recognizer_cnn4.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # -----------------------------------------------------------------------------
    # CNN model2
    model_cnn2, history_cnn2 = train_or_load_model(build_cnn_model2, 'digits', 'digit_recognizer_cnn5.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # -----------------------------------------------------------------------------
    # CNN model3
    model_cnn3, history_cnn3 = train_or_load_model(build_cnn_model3, 'digits', 'digit_recognizer_cnn6.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # -----------------------------------------------------------------------------

    # Prediction
    # Plot the test image from MNIST and the prediction result (each)
    '''
    models_dense_dict = {"Dense1": model_lr, "Dense2": model_mlp}
    models_cnn_dict = {"CNN1": model_cnn1, "CNN2": model_cnn2, "CNN3": model_cnn3}
    idx = 12
    for model_name, model in models_dense_dict.items():
        # Predict
        predictions = predict_image(model, x_test)
        # Get the result
        prediction, confidence, img, true_label = get_prediction_info(predictions, idx, x_test, y_test)

        create_prediction_plot(model_name, predictions[idx], prediction, confidence, true_label, img)

    for model_name, model in models_cnn_dict.items():
        predictions = predict_image(model, x_test)
        prediction, confidence, img, true_label = get_prediction_info(predictions, idx, x_test, y_test)
        create_prediction_plot(model_name, predictions[idx], prediction, confidence, true_label, img)
    '''
    plot_mode = "combined"
    models_dict = {"Dense1": model_lr, "Dense2": model_mlp, "CNN1": model_cnn1, "CNN2": model_cnn2, "CNN3": model_cnn3}

    # Plot the test image created by myself and the prediction result
    if plot_mode == "combined":
        for i in range(10):
            img_path = f'../tf_practice/digits/test_images/{i}.png'
            true_label = i

            # Load and preprocess the image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_data = preprocess_image(img)
            img_data_cnn = reshape_for_cnn(img_data)

            # Gather predictions from all models
            predictions_dict = {}
            for model_name, model in models_dict.items():
                # Predict
                if 'CNN' in model_name:
                    predictions = predict_image(model, img_data_cnn)
                else:
                    predictions = predict_image(model, img_data)

                # Get the result
                prediction, confidence, _, _ = get_prediction_info(predictions, idx=0)

                # Store predictions and results
                predictions_dict[model_name] = (predictions[0], prediction, confidence)

            # Create the combined plot
            create_combined_prediction_plot(img, predictions_dict, true_label, img_title=f"Custom Image {i}")
    else:
        for i in range(10):
            img_path = f'../tf_practice/digits/test_images/{i}.png'
            true_label = i
            # Load and preprocess the image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = preprocess_image(img)

            for model_name, model in models_dict.items():
                # Reshape the data if the model is CNN
                img_array = reshape_for_cnn(img_array) if 'CNN' in model_name else img_array

                # Predict
                predictions = predict_image(model, img_array)

                # Get the result
                prediction, confidence, _, _ = get_prediction_info(predictions, idx=0)

                # Create the plot
                create_prediction_plot(model_name, predictions[0], prediction, confidence, true_label, img)

    # Display plots
    plt.show()


if __name__ == '__main__':
    main()
