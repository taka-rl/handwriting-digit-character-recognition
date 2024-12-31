# handwriting-digit-character-recognition
A Flask web app for handwriting digit and character recognition using machine learning.  
Although this project is currently under development, the digit recognition system has been developed so far!  
The character recognition system will be developed and merged later.  

## digit recognition 
In the digit recognition, there are two ways to predict, drawing digits or importing images. 
The prediction for drawing digits have been completed and for importing images will be developed later.  

## model for the digit recognition
The CNN model is used with the following layers. 
```
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
        layers.Dense(10, activation='softmax')
    ])
```
The model was trained with MNIST dataset.
```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.fit(*train_data, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=False)
```
The model was compiled with the following settings: 
```
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## How to use
1. Run the main.py  
2. Click the "Draw on Canvas"   
![image](https://github.com/user-attachments/assets/7ee5c965-7877-4359-9029-0be973a29338)
3. Draw any digits from 0 to 9 and click the "Predict" button.  
![image](https://github.com/user-attachments/assets/4ab172c9-5a31-4093-891a-49ea080db0dd)
