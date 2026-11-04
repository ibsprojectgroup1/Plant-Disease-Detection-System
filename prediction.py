
import os
import json
import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

from preprocessing_code import (
    ResNet50Preprocessing,
    prepare_convnext_input,
    prepare_mobilenetv2_input
)


# =====================================
# 1. Project Paths and Class Mappings
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "trained_models_valid")

with open(os.path.join(BASE_DIR, "class_mapping.json"), "r") as f:
    CLASS_MAPPING = json.load(f)


# =====================================
# 2. Load All Five Models
# =====================================

def load_models():

    models = {}

    # General leaf validator
    models["leaf_validator"] = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "Leaf_Validator_MobileNetV2.keras"),
        compile=False
    )

    # Cassava validator
    models["cassava_validator"] = tf.keras.models.load_model(
        os.path.join(
            MODEL_DIR,
            "Cassava_Validator_MobileNetV2_Improved.keras"
        ),
        compile=False
    )

    # Maize validator
    models["maize_validator"] = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "Maize_Validator_MobileNetV2.keras"),
        compile=False
    )

    # Cassava disease classifier
    models["cassava_classifier"] = tf.keras.models.load_model(
        os.path.join(
            MODEL_DIR,
            "Copy of ConvNeXtTiny_finetuned_best.keras"
        ),
        compile=False
    )

    # Maize disease classifier
    models["maize_classifier"] = tf.keras.models.load_model(
        os.path.join(
            MODEL_DIR,
            "Copy of ResNet50_finetuned_best.keras"
        ),
        custom_objects={
            "preprocess_input": resnet_preprocess_input,
            "ResNet50Preprocessing": ResNet50Preprocessing,
            "Custom>ResNet50Preprocessing": ResNet50Preprocessing
        },
        compile=False
    )

    return models


# =====================================
# 3. Main Prediction Pipeline
# =====================================

def predict_leaf(image_file, crop_type, models=None):

    crop_type = crop_type.lower()

    if crop_type not in ["cassava", "maize"]:
        raise ValueError(
            "crop_type must be either 'cassava' or 'maize'."
        )

    if models is None:
        models = load_models()

    image = Image.open(image_file).convert("RGB")
    image = np.array(image, dtype=np.float32)

    image_tensor = tf.expand_dims(image, axis=0)


    # =================================
    # Stage 1: General Leaf Validation
    # =================================

    leaf_input = prepare_mobilenetv2_input(image_tensor)

    not_leaf_probability = float(
        models["leaf_validator"].predict(
            leaf_input, verbose=0
        )[0][0]
    )

    if not_leaf_probability >= 0.5:

        return {
            "is_leaf": False,
            "is_correct_crop": False,
            "message": (
                "We couldn't identify this as a valid leaf image. "
                "Please try uploading a clearer, well-lit leaf image."
            ),
            "confidence": not_leaf_probability
        }


    # =================================
    # Stage 2: Crop-Specific Validation
    # =================================

    validator = models[f"{crop_type}_validator"]

    crop_input = prepare_mobilenetv2_input(image_tensor)

    not_crop_probability = float(
        validator.predict(
            crop_input, verbose=0
        )[0][0]
    )

    if not_crop_probability >= 0.5:

        return {
            "is_leaf": True,
            "is_correct_crop": False,
            "crop_type": crop_type,
            "message": (
                f"This image does not appear to be a {crop_type} leaf. "
                f"Please upload a clear {crop_type} leaf image."
            ),
            "confidence": not_crop_probability
        }


    # =================================
    # Stage 3: Disease Classification
    # =================================

    classifier = models[f"{crop_type}_classifier"]

    if crop_type == "cassava":

        classifier_input = prepare_convnext_input(image_tensor)

    else:

        # ResNet50 contains its own preprocessing layer
        classifier_input = tf.image.resize(
            image_tensor, (224, 224)
        )


    probabilities = classifier.predict(
        classifier_input, verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_MAPPING[crop_type][
        str(predicted_index)
    ]

    confidence = float(probabilities[predicted_index])


    # =================================
    # Stage 4: Return Prediction
    # =================================

    return {
        "is_leaf": True,
        "is_correct_crop": True,
        "crop_type": crop_type,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": {
            CLASS_MAPPING[crop_type][str(i)]: float(probabilities[i])
            for i in range(len(probabilities))
        }
    }
