"""Image augmentation pipeline for aerospace defect detection training."""

import tensorflow as tf


def build_augmentation_pipeline(
    random_flip: bool = True,
    random_rotation: bool = True,
    random_brightness: bool = True,
    random_contrast: bool = True,
    rotation_factor: float = 0.1,
    brightness_factor: float = 0.1,
    contrast_lower: float = 0.9,
    contrast_upper: float = 1.1,
) -> tf.keras.Sequential:
    """Build a Keras augmentation pipeline for training images.

    Augmentation is applied only during training — not during validation
    or inference. This is enforced by passing training=True/False to the
    pipeline call.

    Args:
        random_flip: Apply random horizontal and vertical flips.
        random_rotation: Apply random rotation.
        random_brightness: Apply random brightness adjustment.
        random_contrast: Apply random contrast adjustment.
        rotation_factor: Max rotation as fraction of 2π.
        brightness_factor: Max brightness delta as fraction of max value.
        contrast_lower: Lower bound for contrast factor.
        contrast_upper: Upper bound for contrast factor.

    Returns:
        Keras Sequential model containing augmentation layers.
    """
    layers = []

    if random_flip:
        layers.append(tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"))
    if random_rotation:
        layers.append(tf.keras.layers.RandomRotation(factor=rotation_factor))
    if random_brightness:
        layers.append(tf.keras.layers.RandomBrightness(factor=brightness_factor))
    if random_contrast:
        layers.append(
            tf.keras.layers.RandomContrast(factor=(contrast_lower, contrast_upper))
        )

    return tf.keras.Sequential(layers, name="augmentation_pipeline")


def apply_augmentation(
    image: tf.Tensor,
    augmentation_pipeline: tf.keras.Sequential,
    training: bool = True,
) -> tf.Tensor:
    """Apply augmentation pipeline to a single image tensor.

    Args:
        image: Input image tensor of shape (H, W, C).
        augmentation_pipeline: Compiled augmentation pipeline.
        training: If False, augmentation is skipped (inference mode).

    Returns:
        Augmented image tensor.
    """
    return augmentation_pipeline(image, training=training)
