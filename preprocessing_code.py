
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================
# 1. ResNet50 Custom Preprocessing Layer
# =====================================

@tf.keras.utils.register_keras_serializable()
class ResNet50Preprocessing(layers.Layer):

    def call(self, inputs):

        # Convert [-1, 1] to [0, 255]
        x = (inputs + 1.0) * 127.5

        # Convert RGB to BGR
        x = tf.reverse(x, axis=[-1])

        # Subtract ImageNet channel means
        means = tf.constant(
            [103.939, 116.779, 123.680],
            dtype=x.dtype
        )

        return x - means


# =====================================
# 2. ConvNeXtTiny Preprocessing
# =====================================

def prepare_convnext_input(images):

    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32)

    return images


# =====================================
# 3. ResNet50 Preprocessing
# =====================================

def prepare_resnet50_input(images):

    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32)

    return (images / 127.5) - 1.0


# =====================================
# 4. MobileNetV2 Preprocessing
# =====================================

def prepare_mobilenetv2_input(images):

    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32)

    return preprocess_input(images)
