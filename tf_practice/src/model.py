import tensorflow as tf
from tensorflow.keras import models
from emnist import extract_training_samples, extract_test_samples
import json
import numpy as np
from tf_practice.src.build_models import build_dense_model2, build_cnn_model3


class Model:
    def __init__(self, dataset_name: str,
                 model_type: str,
                 epochs: int = 10,
                 batch_size: int = 128,
                 model_builder=None) -> None:
        self.dataset_name = dataset_name
        self.model_builder = model_builder
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.train_data = None
        self.test_data = None

    def load_dataset(self) -> None:
        """
        Load the MNIST or EMNIST dataset based on dataset_class
        Regarding the EMNIST dataset, these 6 types are available.
            ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
        Although 'byclass' and 'letters' can be used, the rest of the types has not been checked yet.
        """
        if self.dataset_name == 'mnist':
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            self.train_data = (x_train, y_train)
            self.test_data = (x_test, y_test)

        elif self.dataset_name.startswith('emnist'):
            dataset_class = self.dataset_name.split('/')[1]
            x_train, y_train = extract_training_samples(dataset_class)
            x_test, y_test = extract_test_samples(dataset_class)

            self.train_data = (x_train, y_train)
            self.test_data = (x_test, y_test)
        else:
            raise ValueError('Invalid dataset_name. Use mnist or emnist/byclass or emnist/letters')

    def build_model(self) -> None:
        num_classes = self._get_num_classes()
        if self.model_builder:
            self.model = self.model_builder(num_classes=num_classes)
        else:
            if self.model_type == 'CNN':
                self.model = build_cnn_model3(num_classes=num_classes)
            elif self.model_type == 'DNN':
                self.model = build_dense_model2(num_classes=num_classes)

    def _get_num_classes(self) -> int:
        if self.dataset_name == 'mnist':
            return 10
        elif self.dataset_name == 'emnist/letters':
            return 26
        elif self.dataset_name == 'emnist/byclass':
            return 52
        else:
            raise ValueError('Cannot determine num_classes for dataset')

    def compile_model(self) -> None:
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        """
        Train the model based on the dataset type

        `validation_split` is only supported for Tensors or NumPy arrays

        Returns:
            Training history
        """
        history = self.model.fit(*self.train_data,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=0.2,
                                 verbose=False)
        return history

    def load_model(self, model_path: str) -> None:
        self.model = models.load_model(model_path, compile=False)

    def save_model(self, model_path: str) -> None:
        self.model.save(model_path)

    @staticmethod
    def save_model_json(model_path, history) -> None:
        with open(model_path + '.json', 'w') as f:
            json.dump(history.history, f)

    def evaluate_model(self):
        """
        Evaluate the model based on the dataset type
        Returns:
            Evaluation results (loss and accuracy)
        """
        if isinstance(self.test_data, tf.data.Dataset):
            evaluation_loss, evaluation_acc = self.model.evaluate(self.test_data)
        else:
            # x_test, y_test = self.test_data or *self.test_data
            evaluation_loss, evaluation_acc = self.model.evaluate(*self.test_data)

        return evaluation_loss, evaluation_acc

    def predict_data(self, data: np.ndarray) -> np.ndarray:
        """
        Predict the input data.

        Parameters:
            data: input data

        Returns:
            Prediction result
        """
        if self.model is None:
            print('The model is not loaded yet. Please load a model first')
        else:
            return self.model.predict(data)
