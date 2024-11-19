"""
・links
https://www.youtube.com/watch?v=eU0FFjYumCI
https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial
https://www.tensorflow.org/datasets/keras_example

"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# Load MNIST for digit recognition
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalization
x_train = x_train / 255
x_test = x_test / 255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

# plt.imshow(x_train[0], cmap='Greys')
# plt.show()

#
model_lr = tf.keras.models.Sequential([
    layers.Input(x_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
model_lr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lr.summary()

model_mlp = tf.keras.models.Sequential([
    layers.Input(x_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(10, activation='softmax')
])
model_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_mlp.summary()

# Train the model
# y_onehot_train = tf.one_hot(y_train, 10)  # if you use loss='categorical_crossentropy
history_lr = model_lr.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=[x_test, y_test], verbose=False)  # validation_data=[x_test, y_test] or validation_split=0.2

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

probs = model_lr.predict(x_test[:5])
preds = np.argmax(probs, axis=1)
for i in range(0, 5):
    print(probs[i], " => ", preds[i])
    plt.imshow(x_test[i], cmap="Greys")
    plt.figure(i)
plt.show()


