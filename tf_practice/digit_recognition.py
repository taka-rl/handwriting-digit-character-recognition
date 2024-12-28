from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from build_models import (train_or_load_model, build_dense_model1,
                          build_dense_model2, build_cnn_model1, build_cnn_model2, build_cnn_model3)


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


def create_prediction_plot(model_name: str,
                           predictions: np.ndarray,
                           prediction: int,
                           confidence: float,
                           true_label: int,
                           img: Image.Image) -> None:
    """
    Create a prediction plot including the specific input image placed on the left side of the plot,
    the prediction result and the distribution bar chart placed on the right sice of the plot.

    Parameters:
        model_name (str): The name of the model.
        predictions:
        prediction:
        confidence:
        true_label:
        img
    """
    # Set the plot size
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    fig.suptitle(f'Input image and prediction result', fontsize=16)

    # Create a plot for the input image
    axes[0].set_title('Input Image', fontsize=10)
    axes[0].grid(False)
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')

    # Create a plot for the prediction result
    axes[1].set_title(f'{model_name}: Prediction Distribution\n'
                      f'Prediction: {prediction}, {confidence:.2f}% (True: {true_label})', fontsize=12)

    # Display the prediction bar chart
    axes[1].grid(False)
    axes[1].bar(range(10), predictions, width=0.4, color="#777777")
    axes[1].set_ylim([0, 1])

    # Highlight the bar chart with blue if the prediction is correct
    axes[1].patches[prediction].set_facecolor('red')
    if true_label == prediction:
        axes[1].patches[prediction].set_facecolor('blue')

    # Adjust the plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))


def create_combined_prediction_plot(img: Image.Image,
                                    predictions_dict: dict,
                                    true_label: int,
                                    img_title: str = "Input Image") -> None:
    """
    Create a combined plot for predictions from multiple models for a single input image.

    Parameters:
        img (Image.Image): The input image.
        predictions_dict (dict): A dictionary with model names as keys and tuples of
                                 (predictions, predicted label, confidence) as values.
        true_label (int): The true label of the image.
        img_title (str): Title for the input image plot.
    """
    # Set the plot size
    num_models = len(predictions_dict)
    fig, axes = plt.subplots(nrows=1, ncols=num_models+1, figsize=(18, 4))
    fig.suptitle(f"Predictions for {img_title}", fontsize=12)

    # Plot the input image on the top-left
    axes[0].set_title(img_title, fontsize=10)
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')

    # Plot predictions for each model
    for i, (model_name, (predictions, predicted_label, confidence)) in enumerate(predictions_dict.items(), start=1):
        # Display the model name and prediction
        axes[i].set_title(f"{model_name}\nPrediction: {predicted_label}, {confidence:.2f}%\nTrue: {true_label}")

        # Display the prediction bar chart
        axes[i].bar(range(10), predictions, width=0.4, color="#777777")
        axes[i].set_ylim([0, 1])
        axes[i].grid(False)

        # Ensure all x-axis labels are displayed
        axes[i].set_xticks(range(10))  # Explicitly set tick positions
        axes[i].set_xticklabels(range(10))  # Explicitly set tick labels

        # Highlight the predicted label
        axes[i].patches[predicted_label].set_facecolor('red')
        if true_label == predicted_label:
            axes[i].patches[predicted_label].set_facecolor('blue')

    # Adjust the plot
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Save the plot
    # plt.savefig(f'../tf_practice/digits/test_results/result_{true_label}.png', dpi='figure', format=None)


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
    model_lr, history_lr = train_or_load_model(build_dense_model1, 'digit_recognizer_dense1.h5',
                                               (x_train, y_train), (x_test, y_test))
    # -----------------------------------------------------------------------------
    # dense model2
    model_mlp, history_mlp = train_or_load_model(build_dense_model2, 'digit_recognizer_dense2.h5',
                                                 (x_train, y_train),(x_test, y_test))
    # -----------------------------------------------------------------------------
    # CNN model1
    model_cnn1, history_cnn1 = train_or_load_model(build_cnn_model1, 'digit_recognizer_cnn1.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # -----------------------------------------------------------------------------
    # CNN model2
    model_cnn2, history_cnn2 = train_or_load_model(build_cnn_model2, 'digit_recognizer_cnn2.h5',
                                                   (x_train_cnn, y_train), (x_test_cnn, y_test))
    # -----------------------------------------------------------------------------
    # CNN model3
    model_cnn3, history_cnn3 = train_or_load_model(build_cnn_model3, 'digit_recognizer_cnn3.h5',
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
