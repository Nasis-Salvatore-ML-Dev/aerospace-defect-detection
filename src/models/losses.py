"""
Custom Loss Functions
src/models/losses.py

Purpose: Define the severity-weighted categorical cross-entropy loss used
to train the aerospace defect detector.

Why a custom loss?
------------------
Standard categorical cross-entropy treats all misclassifications equally.
In aerospace inspection, this is wrong:

    False negative (missed defect) → component ships with defect → potential
    structural failure, injury, regulatory violation. Cost: catastrophic.

    False positive (flagged clean part) → component held for manual review.
    Cost: minor delay.

The asymmetry demands a loss function that penalises false negatives more
heavily than false positives.  We achieve this by assigning a higher weight
to defect classes during loss computation.

Implementation
--------------
We subclass keras.losses.Loss so the custom loss:
    1. Integrates cleanly with model.compile(loss=...).
    2. Is serialisable — model.save() preserves the loss configuration.
    3. Can be loaded back with keras.models.load_model() without custom
       object registration (get_config() / from_config() handles this).

Two loss variants are provided:
    SeverityWeightedCrossEntropy — class-weighted softmax cross-entropy.
    FocalLoss                    — focal loss for extreme class imbalance.

Interview talking points
------------------------
"I implemented a custom Keras loss by subclassing keras.losses.Loss.
The call() method receives y_true (integer class indices) and y_pred
(softmax probabilities).  I gather the per-sample loss from standard
sparse cross-entropy, then multiply each sample's loss by its class
weight before reducing to a scalar mean.  This forces the optimiser to
update weights more aggressively when it misses a defect class."
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Severity-Weighted Cross-Entropy
# ---------------------------------------------------------------------------


class SeverityWeightedCrossEntropy(keras.losses.Loss):
    """
    Categorical cross-entropy with per-class severity weights.

    The loss for each sample is multiplied by the weight of its true class
    before averaging across the batch.  Defect classes receive a higher
    weight than the "good" class, forcing the model to prioritise recall
    on defect samples.

    Mathematical form:
        L = mean( w[y_true] * CE(y_true, y_pred) )

    where CE is standard sparse categorical cross-entropy and w[c] is the
    weight for class c.

    Args:
        class_weights:  List of per-class weights, ordered by class index.
                        Index 0 = "good", indices 1+ = defect classes.
                        Example: [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        name:           Loss name shown in model.summary() and logs.

    Example:
        loss = SeverityWeightedCrossEntropy(
            class_weights=[1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        )
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    """

    def __init__(
        self,
        class_weights: list[float],
        name: str = "severity_weighted_cross_entropy",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights
        # Convert to a constant tensor for efficient gather operations
        self._weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute severity-weighted cross-entropy loss.

        Args:
            y_true: Integer class indices, shape (batch_size,) or (batch_size, 1).
            y_pred: Softmax probabilities,  shape (batch_size, num_classes).

        Returns:
            Scalar mean loss over the batch.
        """
        # Ensure y_true is int32 and flat — shape (batch_size,)
        y_true = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int32)

        # Standard sparse categorical cross-entropy, per sample
        # shape: (batch_size,)
        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )

        # Gather the weight for each sample's true class
        # tf.gather picks self._weights_tensor[y_true[i]] for each i
        # shape: (batch_size,)
        sample_weights = tf.gather(self._weights_tensor, y_true)

        # Weighted loss: multiply each sample's loss by its class weight
        weighted_loss = per_sample_loss * sample_weights

        # Return the mean over the batch (scalar)
        return tf.reduce_mean(weighted_loss)

    def get_config(self) -> dict:
        """
        Return serialisable config so the loss survives model.save() / load_model().

        Without get_config(), Keras cannot reconstruct the loss object when
        loading a SavedModel that was compiled with this custom loss.
        """
        config = super().get_config()
        config.update({"class_weights": self.class_weights})
        return config

    @classmethod
    def from_config(cls, config: dict) -> "SeverityWeightedCrossEntropy":
        """Reconstruct the loss from a saved config dict."""
        return cls(**config)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(keras.losses.Loss):
    """
    Focal loss for extreme class imbalance (Lin et al., 2017).

    Standard cross-entropy assigns equal weight to easy and hard examples.
    In a dataset where 90%+ samples are "good", the model quickly learns to
    predict "good" for everything and achieves high accuracy while missing
    all defects.

    Focal loss down-weights easy examples (high confidence, correct class)
    and focuses training on hard examples (low confidence or wrong class).

    Mathematical form:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t    = model probability for the true class
        gamma  = focusing parameter (0 = standard CE, 2 is typical)
        alpha  = per-class weight (same role as class_weights above)

    Args:
        gamma:         Focusing parameter. Higher = more focus on hard examples.
                       Typical values: 1.0, 2.0 (default), 5.0.
        alpha:         Per-class weight list. Same convention as
                       SeverityWeightedCrossEntropy.class_weights.
        name:          Loss name.

    Reference:
        Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
        https://arxiv.org/abs/1708.02002

    Interview note:
        "Focal loss was introduced for object detection (RetinaNet) but works
        well for any severely imbalanced classification task.  The (1-p_t)^gamma
        term acts as a modulating factor: when the model predicts the correct
        class with high confidence, p_t is close to 1, so (1-p_t)^gamma ≈ 0
        and the loss contribution is negligible.  When the model is wrong or
        uncertain, p_t is low, (1-p_t)^gamma ≈ 1, and the loss is close to
        standard cross-entropy."
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        name: str = "focal_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self._alpha_tensor = (
            tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.

        Args:
            y_true: Integer class indices, shape (batch_size,).
            y_pred: Softmax probabilities,  shape (batch_size, num_classes).

        Returns:
            Scalar mean focal loss over the batch.
        """
        y_true = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int32)

        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # p_t: probability assigned to the true class for each sample
        # shape: (batch_size,)
        batch_size = tf.shape(y_true)[0]
        indices = tf.stack(
            [tf.range(batch_size), y_true], axis=1
        )  # shape: (batch_size, 2)
        p_t = tf.gather_nd(y_pred, indices)  # shape: (batch_size,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Cross-entropy for the true class: -log(p_t)
        ce = -tf.math.log(p_t)

        # Apply alpha (per-class weights) if provided
        if self._alpha_tensor is not None:
            alpha_t = tf.gather(self._alpha_tensor, y_true)
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce

        return tf.reduce_mean(focal_loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config: dict) -> "FocalLoss":
        return cls(**config)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

# Default class weights for the aerospace defect detection task.
# Index 0 = "good" (weight 1.0), indices 1-7 = defect classes (weight 10.0).
# These match the class_weight dict in train.py.
DEFAULT_CLASS_WEIGHTS = [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]


def build_loss(
    loss_type: str = "severity_weighted",
    class_weights: list[float] | None = None,
    focal_gamma: float = 2.0,
) -> keras.losses.Loss:
    """
    Factory function: return the appropriate loss object by name.

    Args:
        loss_type:      "severity_weighted" or "focal".
        class_weights:  Per-class weights. Defaults to DEFAULT_CLASS_WEIGHTS.
        focal_gamma:    Focusing parameter for focal loss (ignored otherwise).

    Returns:
        Instantiated keras.losses.Loss subclass.

    Example:
        loss = build_loss("severity_weighted")
        model.compile(optimizer="adam", loss=loss)
    """
    weights = class_weights or DEFAULT_CLASS_WEIGHTS

    if loss_type == "severity_weighted":
        return SeverityWeightedCrossEntropy(class_weights=weights)
    elif loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=weights)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Choose 'severity_weighted' or 'focal'."
        )
