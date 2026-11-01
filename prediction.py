

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from preprocessing_code import (
    ResNet50Preprocessing,
    prepare_resnet50_input,
    prepare_efficientnetb0_input,
    prepare_mobilenetv2_input
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")

with open(os.path.join(BASE_DIR, "class_mapping.json"), "r") as f:
    CLASS_MAPPING = json.load(f)


# =====================================
# 1. Load Disease Classification Models
# =====================================

def load_model(model_type="ResNet50"):

    if model_type == "ResNet50":
        model_path = os.path.join(
            MODEL_DIR, "ResNet50_V2_Final.keras"
        )

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ResNet50Preprocessing": ResNet50Preprocessing,
                "Custom>ResNet50Preprocessing": ResNet50Preprocessing
            },
            compile=False
        )

    elif model_type == "EfficientNetB0":
        model_path = os.path.join(
            MODEL_DIR, "EfficientNetB0_V2_Final.keras"
        )

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    else:
        raise ValueError("Unsupported model type.")

    return model


# =====================================
# 2. Load MobileNetV2 Leaf Validator
# =====================================

def load_leaf_validator():

    model_path = os.path.join(
        MODEL_DIR, "Leaf_Validator_MobileNetV2.keras"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# =====================================
# 3. Predict Leaf Disease
# =====================================

def predict_leaf(
    image_file,
    model_type="ResNet50",
    model=None,
    validator=None
):

    # Load models if they were not supplied
    if validator is None:
        validator = load_leaf_validator()

    if model is None:
        model = load_model(model_type)

    # Open uploaded image
    image = Image.open(image_file).convert("RGB")
    image = np.array(image, dtype=np.float32)

    # Prepare image for MobileNetV2
    validator_input = tf.expand_dims(image, axis=0)
    validator_input = prepare_mobilenetv2_input(validator_input)

    # Predict whether image is NOT a leaf
    not_leaf_probability = float(
        validator.predict(validator_input, verbose=0)[0][0]
    )

    # Reject image if it is not a leaf
    if not_leaf_probability >= 0.5:

        return {
            "is_leaf": False,
            "message": (
                "We couldn't identify this as a valid leaf image. "
                "Please try uploading a clearer, well-lit image of a "
                "cassava or maize leaf."
            ),
            "confidence": not_leaf_probability
        }

    # =================================
    # Continue with Disease Prediction
    # =================================

    image = tf.expand_dims(image, axis=0)

    if model_type == "ResNet50":
        image = prepare_resnet50_input(image)

    elif model_type == "EfficientNetB0":
        image = prepare_efficientnetb0_input(image)

    probabilities = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_MAPPING[str(predicted_index)]
    confidence = float(probabilities[predicted_index])

    return {
        "is_leaf": True,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": {
            CLASS_MAPPING[str(i)]: float(probabilities[i])
            for i in range(len(probabilities))
        }
    }
