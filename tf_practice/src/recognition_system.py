from PIL import Image
import numpy as np


class RecognitionSystem:
    def __init__(self):
        pass

    @staticmethod
    def preprocess_mnist(dataset: tuple[np.ndarray, np.ndarray]):
        """
        Preprocess the MNIST dataset: normalize images.

        Parameters:
            dataset: tuple[np.ndarray, np.ndarray] -> (x_train, y_train)
                Ex. (x_train, y_train), (x_test, y_test) = mnist.load_data()

        Returns:
            tuple[np.ndarray, np.ndarray]: Normalized dataset with each label
        """
        # Split dataset into data and label
        data, label = dataset
        # Normalization uint8 -> float32
        return data.astype('float32') / 255.0, label

    @staticmethod
    def preprocess_emnist_letters(dataset: tuple[np.ndarray, np.ndarray]):
        """
        Preprocess EMNIST letters data by normalizing images and shifting labels from [1, 26] to [0, 25].

        Parameters:
            dataset: tuple[np.ndarray, np.ndarray] -> (images, labels)

        Returns:
            tuple[np.ndarray, np.ndarray]: Normalized dataset with shifted labels
        """
        # Split dataset into data and label
        data, label = dataset

        # Create a writable copy of the label array
        label = np.copy(label)
        np.subtract(label, 1, out=label)

        # Normalization uint8 -> float32
        return data.astype('float32') / 255.0, label

    @staticmethod
    def preprocess_emnist_byclass(dataset: tuple[np.ndarray, np.ndarray]):
        """
        Preprocess EMNIST byclass data by normalizing images,
         and extracting, filtering labels for only uppercase and lowercase letters.

        In the byclass dataset:
            Labels from 0 to 9: Digits
            Labels from 10 to 35: Uppercase letters
            Labels from 36 to 61: Lowercase letters

        Parameters:
            dataset: tuple[np.ndarray, np.ndarray] -> (images, labels)

        Returns:
            tuple[np.ndarray, np.ndarray]: Normalized and extracted dataset
                                            including only uppercase and lowercase letters.
        """
        # Split dataset into data and label
        data, label = dataset

        # Filter to only the letter samples
        letter_indices = np.where((label >= 10) & (label <= 61))[0]

        # Filter images and labels
        data = data[letter_indices]
        label = label[letter_indices]

        # Adjust labels to range
        label = label - 10.

        # Normalization uint8 -> float32
        return data.astype('float32') / 255.0, label

    @staticmethod
    def preprocess_image(img: Image.Image, target_size: tuple[int, int] = (28, 28)) -> np.ndarray:
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

    @staticmethod
    def reshape_for_cnn(dataset: np.ndarray) -> np.ndarray:
        """
        Reshape the data for CNN models

        Parameters:
            dataset: data to reshape

        Returns:
            np.ndarray: The reshaped data for CNN models
        """
        # Split dataset into data and label
        data, label = dataset
        return data.reshape(data.shape + (1,)), label

    @staticmethod
    def import_image(img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        return RecognitionSystem.preprocess_image(img)

    @staticmethod
    def get_prediction_info(predictions: np.ndarray,
                            idx: int = 0,
                            x_test: np.ndarray = None,
                            y_test: np.ndarray = None):
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
