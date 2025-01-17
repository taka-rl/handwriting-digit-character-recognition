# handwriting-digit-character-recognition
A Flask web app for handwriting digit and character recognition using machine learning.  
Although this project is currently under development, both digit and character recognition systems have been developed so far!  
Currently digit and character recognition systems have been implemented in separate routes(submit-digit, submit-character, upload-digit, upload-character). However, they will be in a single route based on functionalities(drawing or importing) later. For example, submit, upload.

## digit and character recognition systems
There are two ways to predict, drawing or importing images. 
The prediction for drawing digits and characters have been completed.
The frontend of prediction for importing images will be developed later.

## model for both digit and character recognition
The CNN model is used with the following layers. 
```
    num_classes is 10 for the digit recognition and 52 for the character recognition.
    model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
```
The model for the digit recognition system was trained with MNIST dataset.
```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.fit(*train_data, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=False)
```
The model for the character recognition system was trained with EMNIST byclass dataset.
```
x_train, y_train = extract_training_samples(dataset_class)
x_test, y_test = extract_test_samples(dataset_class)
model.fit(*train_data, epochs=10, batch_size=128, validation_split=0.2, verbose=False)
```

The model was compiled with the following settings: 
```
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## How to use
1. Run the main.py  
2. Click either "Draw a Digit on Canvas" or "Draw a Character on Canvas".
3. Draw any digits from 0 to 9 or draw any characters, and click the "Predict" button.  
"Draw a Digit on Canvas"
![image](https://github.com/user-attachments/assets/05f09d5b-25ff-44d4-a07f-841e0d84adc5)
"Draw a Character on Canvas"
![image](https://github.com/user-attachments/assets/6e33bf48-9fc9-4483-a993-34e27265e01a)


## Todo
- Deepen the understanding of TensorFlow and machine learning and deep learning knowledges 
- Improve the model performance and accuracy for both digits and character recognition systems 
- Build a model including both digits and characters
- Make a single route based on functionalities(drawing or importing) such as submit route(drawing), upload route(importing).
- Deploy the app
- Build a way of collecting data input by users and train the model with the collected data to improve
