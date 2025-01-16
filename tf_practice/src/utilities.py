import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json


def plot_training_history_from_json(json_files, labels):
    """
    Plot training history from JSON files.

    Parameters:
        json_files (list): List of file paths to JSON files.
        labels (list): Corresponding labels for the models.
    """
    plt.figure(figsize=(12, 5))
    for json_file, label in zip(json_files, labels):
        with open(json_file, 'r') as f:
            history = json.load(f)

        # Plot training and validation accuracy
        plt.plot(history['val_accuracy'], label=f'{label} Val Accuracy')
        plt.plot(history['accuracy'], '--', label=f'{label} Train Accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.title("Model Training Accuracy Comparison")
    plt.show()


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
