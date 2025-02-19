# handwriting-digit-character-recognition
**This project is currently under development.**  
A Flask web app for handwriting digit and character recognition.  
The functions implemented are as follows: 
- Handwriting drawn Digit/Character recognition
- This app contains a data pipeline. 
  - Feedback mechanism where users can confirm or correct predictions, sending validated data to Google Spreadsheet for data
  collection.
  - Retraining a model with the collected data to enhance accuracy. (Ideally but it isn't easy to collect a lot of data. Thus generated data from the MNIST dataset is used at the moment.)
  - Automatically commit model files and deploy (This hasn't been implemented. Future plan)

- Game mode(This hasn't been implemented. Future plan)

## digit and character recognition systems
There are two ways to predict, drawing or importing images. 
The prediction for drawing digits and characters have been completed.
The frontend of prediction for importing images will be developed later.

## Folder structure
    │── app
    │   ├── routes               # Store Blueprint routes here
    │   │   ├── __init__.py
    │   │   ├── canvas.py        # Handles digit/character drawing
    │   │   ├── import_file.py   # Handles image uploads
    │   │   ├── index.py         # Home route
    │   ├── models.py            # Loads models at app startup
    │   ├── utilities.py         # Helper functions (image processing, validation)
    │   ├── gss.py               # Google Sheets API logic
    │   ├── dummy_data.py        # Generate rotated data from the MNIST data
    │   ├── retrain_model.py     # Retrain a model with the generated data from dummy_data.py
    │   ├── static               # Static files (CSS, JS, models for recognition)
    │   ├── templates            # HTML templates
    │   └──  __init__.py         # Creates the Flask app and registers Blueprints
    │── doc                      # Documents
    │── tests                    # Unit testing
    │── main.py                  # Entry point of the app
    │── requirements.txt         # Dependencies
    │── .gcloudignore            # Ignore sensitive files for deployment on GCP
    │── .gitignore               # Ignore sensitive files
    │── app.yaml                 # Deployment
    └── README.md


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
![image](https://github.com/user-attachments/assets/cb349fcf-0753-457a-84df-599989f02e13)

"Draw a Character on Canvas"
![image](https://github.com/user-attachments/assets/7ea5293c-6bfd-4d1f-a4ea-b4bf6fbd900c)

### Regarding the user feedback
User can give a feedback against the prediction result.  
If either "Yes" or "No" is clicked, the drawn digit/character, prediction results(predict label and confidence) and the correct label are sent to the Google Spreadsheet.
Users can input the correct label if the prediction is not correct. 
![image](https://github.com/user-attachments/assets/4b19a3de-6a56-4008-ae4b-ef4ba0bf445d)
If the data is sent properly, the following message shows up.  
![image](https://github.com/user-attachments/assets/f0a6833f-7a7b-4fde-922a-87741ba5984c)


## Todo
- Deepen the understanding of TensorFlow and machine learning and deep learning knowledges 
- Improve the model performance and accuracy for both digits and character recognition systems 
- Build a model including both digits and characters
- Make a single route based on functionalities(drawing or importing) such as submit route(drawing), upload route(importing).
- Retrain models with the collected data to improve
- Introduce Continuous Deployment(CD)
- Add a game mode
