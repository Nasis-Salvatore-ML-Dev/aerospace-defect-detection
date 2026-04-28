"""Unit tests for src/models/losses.py."""

import pytest
import tensorflow as tf

from src.models.losses import (
    DEFAULT_CLASS_WEIGHTS,
    FocalLoss,
    SeverityWeightedCrossEntropy,
    build_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 8


def make_batch(
    batch_size: int = 4,
    num_classes: int = NUM_CLASSES,
    true_class: int = 1,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Generate a small batch of labels and predictions for testing.

    y_true: integer class indices, shape (batch_size,)
    y_pred: uniform softmax probabilities, shape (batch_size, num_classes)
    """
    y_true = tf.constant([true_class] * batch_size, dtype=tf.int32)
    # Uniform predictions — each class gets equal probability
    y_pred = tf.constant(
        [[1.0 / num_classes] * num_classes] * batch_size,
        dtype=tf.float32,
    )
    return y_true, y_pred


# ---------------------------------------------------------------------------
# SeverityWeightedCrossEntropy
# ---------------------------------------------------------------------------


class TestSeverityWeightedCrossEntropy:
    def test_returns_scalar(self):
        loss_fn = SeverityWeightedCrossEntropy(class_weights=DEFAULT_CLASS_WEIGHTS)
        y_true, y_pred = make_batch()
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        loss_fn = SeverityWeightedCrossEntropy(class_weights=DEFAULT_CLASS_WEIGHTS)
        y_true, y_pred = make_batch()
        loss = loss_fn(y_true, y_pred)
        assert float(loss) > 0.0

    def test_defect_class_loss_higher_than_good_class(self):
        """Defect class (weight=10) should produce higher loss than good (weight=1)."""
        loss_fn = SeverityWeightedCrossEntropy(class_weights=DEFAULT_CLASS_WEIGHTS)

        # Batch where true class is "good" (index 0, weight 1.0)
        y_true_good, y_pred = make_batch(true_class=0)
        loss_good = float(loss_fn(y_true_good, y_pred))

        # Batch where true class is "crack" (index 1, weight 10.0)
        y_true_defect, y_pred = make_batch(true_class=1)
        loss_defect = float(loss_fn(y_true_defect, y_pred))

        assert loss_defect > loss_good

    def test_defect_loss_approximately_10x_good_loss(self):
        """
        With uniform predictions, the loss ratio should be close to 10
        since the only difference is the class weight multiplier.
        """
        loss_fn = SeverityWeightedCrossEntropy(class_weights=DEFAULT_CLASS_WEIGHTS)

        y_true_good, y_pred = make_batch(true_class=0)
        loss_good = float(loss_fn(y_true_good, y_pred))

        y_true_defect, y_pred = make_batch(true_class=1)
        loss_defect = float(loss_fn(y_true_defect, y_pred))

        ratio = loss_defect / loss_good
        assert abs(ratio - 10.0) < 0.1

    def test_perfect_prediction_lower_loss(self):
        """A confident correct prediction should have lower loss than uniform."""
        loss_fn = SeverityWeightedCrossEntropy(class_weights=DEFAULT_CLASS_WEIGHTS)

        # Uniform predictions
        y_true, y_pred_uniform = make_batch(true_class=1)
        loss_uniform = float(loss_fn(y_true, y_pred_uniform))

        # Confident correct predictions
        confident = [0.01] * NUM_CLASSES
        confident[1] = 1.0 - (0.01 * (NUM_CLASSES - 1))
        y_pred_confident = tf.constant([confident] * 4, dtype=tf.float32)
        loss_confident = float(loss_fn(y_true, y_pred_confident))

        assert loss_confident < loss_uniform

    def test_get_config_roundtrip(self):
        """Loss must survive serialisation roundtrip."""
        weights = [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        loss_fn = SeverityWeightedCrossEntropy(class_weights=weights)
        config = loss_fn.get_config()

        reconstructed = SeverityWeightedCrossEntropy.from_config(config)
        assert reconstructed.class_weights == weights

    def test_unit_weights_matches_standard_crossentropy(self):
        """With all weights=1.0, should match standard sparse CE."""
        unit_weights = [1.0] * NUM_CLASSES
        loss_fn = SeverityWeightedCrossEntropy(class_weights=unit_weights)

        y_true, y_pred = make_batch(true_class=2)
        custom_loss = float(loss_fn(y_true, y_pred))

        standard_loss = float(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            .numpy()
            .mean()
        )

        assert abs(custom_loss - standard_loss) < 1e-5


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_returns_scalar(self):
        loss_fn = FocalLoss(gamma=2.0)
        y_true, y_pred = make_batch()
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        loss_fn = FocalLoss(gamma=2.0)
        y_true, y_pred = make_batch()
        loss = loss_fn(y_true, y_pred)
        assert float(loss) > 0.0

    def test_gamma_zero_approaches_standard_ce(self):
        """Focal loss with gamma=0 should equal standard cross-entropy."""
        loss_fn = FocalLoss(gamma=0.0)
        y_true, y_pred = make_batch(true_class=1)
        focal_loss = float(loss_fn(y_true, y_pred))

        standard_loss = float(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            .numpy()
            .mean()
        )
        assert abs(focal_loss - standard_loss) < 1e-4

    def test_higher_gamma_lower_loss_on_easy_examples(self):
        """
        Higher gamma down-weights easy examples (confident correct predictions).
        So for a confident correct prediction, loss with gamma=2 < loss with gamma=0.
        """
        y_true = tf.constant([1, 1, 1, 1], dtype=tf.int32)
        # High confidence on correct class
        confident = [0.01] * NUM_CLASSES
        confident[1] = 1.0 - (0.01 * (NUM_CLASSES - 1))
        y_pred = tf.constant([confident] * 4, dtype=tf.float32)

        loss_gamma0 = float(FocalLoss(gamma=0.0)(y_true, y_pred))
        loss_gamma2 = float(FocalLoss(gamma=2.0)(y_true, y_pred))

        assert loss_gamma2 < loss_gamma0

    def test_get_config_roundtrip(self):
        loss_fn = FocalLoss(gamma=3.0, alpha=DEFAULT_CLASS_WEIGHTS)
        config = loss_fn.get_config()
        reconstructed = FocalLoss.from_config(config)
        assert reconstructed.gamma == 3.0
        assert reconstructed.alpha == DEFAULT_CLASS_WEIGHTS


# ---------------------------------------------------------------------------
# build_loss factory
# ---------------------------------------------------------------------------


class TestBuildLoss:
    def test_severity_weighted(self):
        loss = build_loss("severity_weighted")
        assert isinstance(loss, SeverityWeightedCrossEntropy)

    def test_focal(self):
        loss = build_loss("focal")
        assert isinstance(loss, FocalLoss)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            build_loss("unknown_loss")

    def test_custom_weights_passed_through(self):
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        loss = build_loss("severity_weighted", class_weights=weights)
        assert loss.class_weights == weights
