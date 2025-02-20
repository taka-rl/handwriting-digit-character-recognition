# Handwriting Digit & Character Recognition (Flask + TensorFlow)
🚀 **Currently under development**  
This is a Flask web application, allowing  users to recognize handwriting digit and character recognition using CNN models trained with TensorFlow.

### ✨ **Key Features**
✔️ **Digit & Character Recognition** → Users can draw or upload images for recognition.  
✔️ **Feedback Mechanism** → Users can correct predictions, sending validated data to **Google Spreadsheets** for future training.  
✔️ **Retraining Pipeline** → The model can be retrained using collected user data.  (Ideally but it isn't easy to collect a lot of data. Thus generated data from the MNIST dataset is used at the moment.)  
✔️ **CI/CD Integration (Future Plan)** → Continuous Testing & Deployment planned.  
✔️ **Game Mode (Future Plan)** → A fun challenge-based mode for handwriting recognition.

## 🔍 How Recognition Works
Users can recognize **handwritten digits and characters** in two ways:
1. **Drawing on a Canvas** → (Implemented ✅)
2. **Uploading an Image** → (Planned 🛠)

Currently, **drawing-based recognition is fully functional**, while **image upload recognition is under development**.


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


## 🧠 CNN Model Architecture
Both **digit and character recognition models** use **Convolutional Neural Networks (CNNs)** with the following architecture:
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
### 🏋️ Model Training
- **Digits** → Trained using the **MNIST dataset**.
- **Characters** → Trained using the **EMNIST ByClass dataset**.
```
- Digit
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.fit(*train_data, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=False)
```
```
- Character
x_train, y_train = extract_training_samples(dataset_class)
x_test, y_test = extract_test_samples(dataset_class)
model.fit(*train_data, epochs=10, batch_size=128, validation_split=0.2, verbose=False)
```

### Compile models 
```
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## 🛠 How to Use
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


## 📌 Future Improvements
✔️ **Enhance Model Performance** → Improve accuracy for both digits and characters.  
✔️ **Combine Digit & Character Models** → Create a unified recognition system.  
✔️ **Simplify API Routes** → Merge "Draw" and "Import" functionalities as there are two routes(digit/character).  
✔️ **Enable Model Retraining** → Collect real user data for training.  
✔️ **Deploy with Continuous Deployment (CD)** → Automate model updates.  
✔️ **Introduce Game Mode** → Challenge users to draw characters quickly & accurately.
