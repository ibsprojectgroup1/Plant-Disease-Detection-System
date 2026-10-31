
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class ResNet50Preprocessing(layers.Layer):
    """
    Exact preprocessing layer used by the trained ResNet50 model.
    Input is expected in the MobileNetV2-normalized range [-1, 1].
    """

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


def prepare_resnet50_input(images):
    """
    Resize and normalize RGB images for the ResNet50 model.
    """
    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32)
    return (images / 127.5) - 1.0


def prepare_efficientnetb0_input(images):
    """
    Prepare images for the EfficientNetB0 model.
    The model contains its own input rescaling.
    """
    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32)
    return images
