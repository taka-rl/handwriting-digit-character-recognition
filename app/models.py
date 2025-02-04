from tensorflow.keras.models import model_from_json
import os

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'models')


def load_model(model_name: str):
    """Load a TensorFlow model from JSON and its corresponding weights."""
    model_json_path = os.path.join(MODELS_PATH, f"{model_name}.json")
    model_weights_path = os.path.join(MODELS_PATH, f"{model_name}_weights.h5")

    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    return model


model_digit = load_model("model_digit")
model_character = load_model("model_character")
character_list = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
