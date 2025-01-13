from PIL import Image
import numpy as np


class RecognitionSystem:
    def __init__(self):
        pass

    @staticmethod
    def preprocess_dataset(dataset: tuple[np.ndarray, np.ndarray]):
        """
        Preprocess the dataset: normalize images.

        Parameters:
            dataset: tuple[np.ndarray, np.ndarray] -> (x_train, y_train)
            Ex. (x_train, y_train), (x_test, y_test) = mnist.load_data()

        Returns:
            dataset: Normalized dataset with each label

        """

        # Normalization uint8 -> float32
        dataset, label = dataset
        return dataset.astype('float32') / 255, label

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
    def reshape_for_cnn(data: np.ndarray) -> np.ndarray:
        """
        Reshape the data for CNN models

        Parameters:
            data: data to reshape

        Returns:
            np.ndarray: The reshaped data for CNN models
        """
        return data.reshape(data.shape + (1,))

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
