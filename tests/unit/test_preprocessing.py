"""Unit tests for data pipeline components."""

import numpy as np
import tensorflow as tf

from src.data.augmentation import build_augmentation_pipeline


def test_augmentation_pipeline_builds() -> None:
    """Verify augmentation pipeline constructs without error."""
    pipeline = build_augmentation_pipeline()
    assert pipeline is not None


def test_augmentation_pipeline_output_shape() -> None:
    """Verify augmentation preserves image shape."""
    pipeline = build_augmentation_pipeline()
    dummy_image = tf.random.uniform(shape=(1, 224, 224, 3))
    output = pipeline(dummy_image, training=True)
    assert output.shape == (1, 224, 224, 3)


def test_augmentation_disabled_at_inference() -> None:
    """Verify augmentation is skipped when training=False."""
    pipeline = build_augmentation_pipeline()
    dummy_image = tf.constant(np.random.rand(1, 224, 224, 3), dtype=tf.float32)
    output = pipeline(dummy_image, training=False)
    np.testing.assert_array_almost_equal(dummy_image.numpy(), output.numpy())
