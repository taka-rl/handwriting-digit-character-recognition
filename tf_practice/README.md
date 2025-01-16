# tf_practice
This folder is for learning TensorFlow and building models for the recognition systems.


## Folder structure in tf_practice
    ├── characters                # trained models and thier results, and test images and results for characters
    │   ├── models                # trained models for both letters and byclass
    │   ├── test_images           # self created test images
    │   └── test_results          # test results
    ├── digits                    # trained models and thier results, and test images and results for digits
    │   ├── models                # trained models
    │   ├── test_images           # self created test images
    │   └── test_results          # test results
    ├── doc                       # codes for tic-tac-toe environment
    ├── src                       # codes for tic-tac-toe environment
    │   ├── model.py              # model class
    │   ├── recognition_system.py # recognition system class
    │   └──              
    ├── best_model_digit.h5       # model for the digit recognition system
    ├── build_models.py           # Run the app
    ├── main.py                   # Run the app
    ├── model.py                  # Run the app
    ├── note.txt                  # Run the app
    ├── README.md                 # Run the app
    ├── recognition_system.py                 # Run the app
    ├── character_recognition.py                 # Run the app
    ├── tensorflow_basics.py               # Run the app
    └── README.md

## about TensorFlow
TensorFlow is an open-source platform for machine learning used by a variety of users such as data scientists, software developers and educators[^1].   

## Dataset used for training models
### MNIST for the digit recognition system
MNIST dataset is a collection of 70,000 hand-writing digit images, including 60, 000 training images and 10,000 test images[^2]. 

### EMNIST for the character recognition system
EMNIST dataset, which is Extended MNIST dataset, is a set of handwritten letters and digits in a 28x28 pixel format[^3].

The dataset can be splited in 6 types[^3]:  
- EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
- EMNIST ByMerge: 814,255 characters. 47 unbalanced classes.
- EMNIST Balanced: 131,600 characters. 47 balanced classes.
- EMNIST Letters: 145,600 characters. 26 balanced classes.
- EMNIST Digits: 280,000 characters. 10 balanced classes.
- EMNIST MNIST: 70,000 characters. 10 balanced classes.

The EMNIST ByClass is used to train a model for the character recognition system. 
It contains 62 different classes, including both uppercase and lowercase letters (26 classes each) as well as 10 digits (10 classes).
To use only the uppercase and lowercase letters, they need to be extracted from the dataset.

## References
[^1]: [TensorFlow ](https://www.nvidia.com/en-eu/glossary/tensorflow/)
[^2]: [Get Started with Computer Vision by Building a Digit Recognition Model with Tensorflow](https://medium.com/artificialis/get-started-with-computer-vision-by-building-a-digit-recognition-model-with-tensorflow-b2216823b90a)
[^3]: [Handwritten Character Recognition Web App with EMNIST](https://guptapurav.medium.com/handwritten-character-recognition-web-app-with-emnist-9af77d895a52)
