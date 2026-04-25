"""
Aerospace Defect Detection — Training Script
src/models/train.py

Trains an EfficientNetB0-based defect classifier on the MVTec AD dataset
(aerospace subset: metal_nut, screw, tile, capsule, cable).

Transfer learning protocol
--------------------------
Phase 1 — Head training (10 epochs)
    EfficientNetB0 backbone frozen.
    Only the classification head trains.
    High LR (1e-3) is safe because backbone weights are untouched.

Phase 2 — Fine-tuning (20 epochs)
    Top 30 layers of backbone unfrozen.
    LR reduced 10x (1e-4) to avoid destroying pretrained weights.
    Early stopping on val_recall (patience=5, restore_best_weights=True).

Loss function — Severity-Weighted Cross-Entropy
    False negatives (missed defects) in aerospace = catastrophic failure.
    We penalise the "good" class (index 0) less and defect classes more.
    class_weight = {0: 1.0, 1: 10.0, 2: 10.0, ...}
    This forces the model to prioritise recall over precision.

Metrics
    Primary:   val_recall   (minimise false negatives)
    Secondary: val_accuracy, val_precision, val_AUC

MLOps integration
    - Firestore: logs run_id, hyperparameters, metrics, artifact paths
    - W&B:       real-time training curves (free tier)
    - SavedModel: versioned export to models/saved_model/v<major>.<minor>/
    - Callbacks:  TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

Usage (Google Colab)
--------------------
    # Mount Drive, install deps, clone repo, then:
    !python src/models/train.py \\
        --data-dir data/mvtec_aerospace \\
        --model-version v1.0 \\
        --epochs-head 10 \\
        --epochs-finetune 20 \\
        --batch-size 32 \\
        --unfreeze-layers 30 \\
        --use-wandb

Cost: $0 — runs entirely on Colab free T4 GPU.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # EfficientNetB0 native input size
AUTOTUNE = tf.data.AUTOTUNE

# Defect classes — must match DefectClass enum in src/api/schemas.py
# Index 0 = "good" (no defect).  All other indices = defect types.
CLASS_NAMES = [
    "good",
    "crack",
    "scratch",
    "bent",
    "color",
    "contamination",
    "hole",
    "broken",
]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 defect classifier on MVTec AD."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mvtec_aerospace"),
        help="Root directory with train/ val/ test/ subdirs.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0",
        help="Semantic version for SavedModel export, e.g. v1.0",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=10,
        help="Epochs to train the classification head (backbone frozen).",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=20,
        help="Epochs for fine-tuning (top N backbone layers unfrozen).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size. Reduce to 16 if Colab OOMs.",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=30,
        help="Number of top backbone layers to unfreeze during fine-tuning.",
    )
    parser.add_argument(
        "--learning-rate-head",
        type=float,
        default=1e-3,
        help="Learning rate for head-only training phase.",
    )
    parser.add_argument(
        "--learning-rate-finetune",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning phase (should be 10x lower).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate on the classification head.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Log metrics to Weights & Biases (free tier).",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        help="GCP project ID for Firestore logging. Skipped if None.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def build_augmentation_pipeline() -> keras.Sequential:
    """
    Keras augmentation layers applied only during training.

    Augmentation is baked into the model graph — it runs on GPU
    automatically and is disabled at inference time (training=False).

    Choices rationale:
    - RandomFlip:       Defects can appear on any orientation.
    - RandomRotation:   Small rotations (±10°) simulate real inspection angles.
    - RandomZoom:       Simulates varying camera distance.
    - RandomContrast:   Simulates varying lighting conditions in hangars.
    - RandomBrightness: Same motivation as contrast.

    We avoid aggressive geometric transforms (large rotations, shear) because
    aerospace components have orientation-specific defects (e.g., a bent edge
    on the left side is different from a bent edge on the right).
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(factor=0.1),
            keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            keras.layers.RandomContrast(factor=0.2),
            keras.layers.RandomBrightness(factor=0.1),
        ],
        name="augmentation",
    )


def load_dataset(
    data_dir: Path,
    split: str,
    batch_size: int,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Load images from data_dir/<split>/ using keras image_dataset_from_directory.

    Expected directory structure:
        data/mvtec_aerospace/
            train/
                good/        ← normal components
                crack/       ← defect type 1
                scratch/     ← defect type 2
                ...
            val/
                good/
                crack/
                ...
            test/
                good/
                crack/
                ...

    EfficientNetB0 preprocessing (include_preprocessing=True in the backbone)
    expects pixel values in [0, 255] — do NOT normalise to [0, 1] here.

    Args:
        data_dir:   Root data directory.
        split:      "train", "val", or "test".
        batch_size: Mini-batch size.
        augment:    If True, apply data augmentation (training only).

    Returns:
        tf.data.Dataset yielding (image_batch, label_batch) tuples.
        Images: float32 tensor, shape (batch, 224, 224, 3), values [0, 255].
        Labels: int32 tensor, shape (batch,), values in [0, NUM_CLASSES).
    """
    split_dir = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Data split directory not found: {split_dir}\n"
            f"Download MVTec AD and run scripts/prepare_dataset.py first."
        )

    dataset = keras.utils.image_dataset_from_directory(
        directory=str(split_dir),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=(split == "train"),
        seed=42,
    )

    if augment:
        augmentation = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Class weights (severity-weighted loss)
# ---------------------------------------------------------------------------


def build_class_weights() -> dict[int, float]:
    """
    Assign higher loss weight to defect classes than to "good".

    Rationale: a false negative (missed defect) in aerospace can cause
    catastrophic structural failure. We weight defect classes 10x higher
    than the "good" class to force the model to prioritise recall.

    In practice, class_weight is passed to model.fit() and Keras multiplies
    each sample's loss by the weight of its true class before averaging.

    Returns:
        Dict mapping class index → weight scalar.
        Example: {0: 1.0, 1: 10.0, 2: 10.0, 3: 10.0, ...}
    """
    weights = {}
    for idx, name in enumerate(CLASS_NAMES):
        weights[idx] = 1.0 if name == "good" else 10.0
    logger.info("Class weights: %s", weights)
    return weights


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(dropout: float = 0.3) -> keras.Model:
    """
    Build EfficientNetB0 transfer learning model.

    Architecture:
        Input (224, 224, 3)
            ↓
        EfficientNetB0 backbone (pretrained ImageNet, frozen)
            ↓
        GlobalAveragePooling2D          ← replaces Flatten
            ↓
        BatchNormalization              ← stabilise head inputs
            ↓
        Dense(256, activation="relu")
            ↓
        Dropout(dropout)               ← regularise head
            ↓
        Dense(NUM_CLASSES, activation="softmax")

    Why EfficientNetB0?
        - Best accuracy/parameter ratio at this scale (5.3M params).
        - include_preprocessing=True: handles pixel rescaling internally,
          so the data pipeline does not need to normalise images.
        - Pretrained on ImageNet: edge/texture/shape features transfer
          well to surface defect detection.

    Args:
        dropout: Dropout rate on the classification head [0, 1].

    Returns:
        Uncompiled Keras model with backbone frozen.
    """
    # Input layer
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")

    # Backbone: EfficientNetB0 pretrained on ImageNet
    # include_top=False: exclude the original 1000-class classification head
    # include_preprocessing=True: applies EfficientNet-specific pixel scaling
    # trainable=False: freeze all backbone weights for Phase 1
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        include_preprocessing=True,
    )
    backbone.trainable = False

    # Classification head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dense(256, activation="relu", name="head_dense")(x)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(
        x
    )

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="aerospace_defect_detector"
    )

    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    logger.info(
        "Model built — trainable params: %s / %s (%.1f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100 * trainable_params / total_params,
    )

    return model


def unfreeze_top_layers(model: keras.Model, n_layers: int) -> None:
    """
    Unfreeze the top N layers of the EfficientNetB0 backbone for fine-tuning.

    Why not unfreeze all layers?
    Early layers detect low-level features (edges, textures) that are universal
    — they do not need updating for surface defect detection. Unfreezing them
    would require far more data and compute, and risks destroying the learned
    representations. Top layers detect high-level, task-specific features and
    benefit from fine-tuning on the target domain.

    Args:
        model:    The compiled Keras model.
        n_layers: Number of top backbone layers to unfreeze.
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze all layers except the top N
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    # BatchNorm layers must stay frozen during fine-tuning to preserve
    # the running mean/variance statistics from ImageNet pretraining.
    # Updating BN stats with a small dataset causes instability.
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    newly_trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    logger.info(
        "Unfroze top %d backbone layers — trainable params now: %s",
        n_layers,
        f"{newly_trainable:,}",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def build_callbacks(
    checkpoint_dir: Path,
    log_dir: Path,
    phase: str,
    use_wandb: bool = False,
) -> list:
    """
    Build Keras callbacks for a training phase.

    Callbacks:
        ModelCheckpoint:     Save best model by val_recall.
        EarlyStopping:       Stop if val_recall does not improve for 5 epochs.
        ReduceLROnPlateau:   Halve LR if val_loss stalls for 3 epochs.
        TensorBoard:         Log metrics for visualisation.
        WandbCallback:       (optional) Sync metrics to W&B dashboard.

    Args:
        checkpoint_dir: Directory for ModelCheckpoint saves.
        log_dir:        Directory for TensorBoard logs.
        phase:          "head" or "finetune" — used in file names.
        use_wandb:      Whether to add W&B callback.

    Returns:
        List of keras.callbacks.Callback instances.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"best_{phase}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_recall",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir / phase),
            histogram_freq=0,  # 0 = no weight histograms (saves time)
            update_freq="epoch",
        ),
    ]

    if use_wandb:
        try:
            import wandb as _wandb  # noqa: F401
            from wandb.integration.keras import WandbMetricsLogger

            callbacks.append(WandbMetricsLogger(log_freq="epoch"))
            logger.info("W&B callback attached.")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")

    return callbacks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> dict[str, float]:
    """
    Evaluate the model on the test set and return a metrics dict.

    Returns:
        Dict with keys: test_loss, test_accuracy, test_recall, test_precision, test_auc
    """
    logger.info("Evaluating on test set ...")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    logger.info("Test results: %s", results)
    return {f"test_{k}": v for k, v in results.items()}


# ---------------------------------------------------------------------------
# SavedModel export
# ---------------------------------------------------------------------------


def export_savedmodel(model: keras.Model, version: str, output_dir: Path) -> Path:
    """
    Export the trained model in TensorFlow SavedModel format.

    Versioned directory structure:
        models/saved_model/
            v1.0/          ← SavedModel directory
            v1.1/          ← next version after retraining

    The API (src/api/app.py) scans this directory at startup and loads
    the latest version automatically — no manual config required.

    Args:
        model:      Trained Keras model.
        version:    Semantic version string, e.g. "v1.0".
        output_dir: Root output directory (models/saved_model).

    Returns:
        Path to the exported SavedModel directory.
    """
    export_path = output_dir / version
    export_path.mkdir(parents=True, exist_ok=True)
    model.save(str(export_path))
    logger.info("SavedModel exported to %s", export_path)
    return export_path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """
    Full two-phase transfer learning training loop.

    Phase 1: Train head only (backbone frozen).
    Phase 2: Fine-tune top N backbone layers (lower LR).
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Firestore: log run started
    # ------------------------------------------------------------------
    run_id: str | None = None
    firestore_logger = None

    if args.gcp_project:
        try:
            from src.monitoring.firestore_logger import FirestoreLogger

            firestore_logger = FirestoreLogger(project_id=args.gcp_project)
            run_id = firestore_logger.log_run_started(
                model_version=args.model_version,
                hyperparameters={
                    "backbone": "EfficientNetB0",
                    "epochs_head": args.epochs_head,
                    "epochs_finetune": args.epochs_finetune,
                    "batch_size": args.batch_size,
                    "learning_rate_head": args.learning_rate_head,
                    "learning_rate_finetune": args.learning_rate_finetune,
                    "dropout": args.dropout,
                    "unfreeze_layers": args.unfreeze_layers,
                    "image_size": IMAGE_SIZE,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "augmentation": True,
                    "class_weight_defect": 10.0,
                },
                dataset={
                    "source": "MVTec AD — aerospace subset",
                    "classes": CLASS_NAMES,
                    "image_size": list(IMAGE_SIZE),
                },
            )
            logger.info("Firestore run started — run_id=%s", run_id)
        except Exception as exc:
            logger.warning("Firestore logging failed (continuing): %s", exc)

    # ------------------------------------------------------------------
    # W&B initialisation
    # ------------------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb as _wandb

            _wandb.init(
                project="aerospace-defect-detection",
                name=f"run_{args.model_version}",
                config=vars(args),
            )
            logger.info("W&B run initialised.")
        except ImportError:
            logger.warning("wandb not installed — skipping.")

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    logger.info("GPUs available: %d", len(gpus))
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    logger.info("Loading datasets from %s ...", args.data_dir)
    train_ds = load_dataset(args.data_dir, "train", args.batch_size, augment=True)
    val_ds = load_dataset(args.data_dir, "val", args.batch_size, augment=False)
    test_ds = load_dataset(args.data_dir, "test", args.batch_size, augment=False)

    class_weights = build_class_weights()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    model = build_model(dropout=args.dropout)

    # Metrics used in both phases
    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.AUC(name="auc", multi_label=False),
    ]

    # ------------------------------------------------------------------
    # Phase 1: Head training (backbone frozen)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: Head training (%d epochs)", args.epochs_head)
    logger.info("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate_head),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )
    model.summary(print_fn=logger.info)

    checkpoint_dir = Path("models/checkpoints")
    log_dir = Path("logs/tensorboard")

    head_callbacks = build_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        phase="head",
        use_wandb=args.use_wandb,
    )

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=head_callbacks,
        verbose=1,
    )

    logger.info(
        "Phase 1 complete — best val_recall: %.4f",
        max(history_head.history.get("val_recall", [0.0])),
    )

    # ------------------------------------------------------------------
    # Phase 2: Fine-tuning (top N backbone layers unfrozen)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(
        "PHASE 2: Fine-tuning (top %d layers, %d epochs)",
        args.unfreeze_layers,
        args.epochs_finetune,
    )
    logger.info("=" * 60)

    unfreeze_top_layers(model, n_layers=args.unfreeze_layers)

    # Recompile required after changing trainable status
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate_finetune),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )

    finetune_callbacks = build_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        phase="finetune",
        use_wandb=args.use_wandb,
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weights,
        callbacks=finetune_callbacks,
        verbose=1,
    )

    best_val_recall = max(history_finetune.history.get("val_recall", [0.0]))
    logger.info("Phase 2 complete — best val_recall: %.4f", best_val_recall)

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    test_metrics: dict[str, float] = {}
    test_metrics = evaluate_model(model, test_ds)

    # ------------------------------------------------------------------
    # SavedModel export
    # ------------------------------------------------------------------
    savedmodel_dir = Path("models/saved_model")
    export_path = export_savedmodel(model, args.model_version, savedmodel_dir)

    # ------------------------------------------------------------------
    # Firestore: log run completed
    # ------------------------------------------------------------------
    duration = time.time() - start_time

    if firestore_logger and run_id:
        try:
            val_metrics = {
                "val_recall": best_val_recall,
                "val_accuracy": max(
                    history_finetune.history.get("val_accuracy", [0.0])
                ),
                "val_precision": max(
                    history_finetune.history.get("val_precision", [0.0])
                ),
                "val_auc": max(history_finetune.history.get("val_auc", [0.0])),
            }
            if test_metrics:
                val_metrics.update(test_metrics)

            firestore_logger.log_run_completed(
                run_id=run_id,
                metrics=val_metrics,
                artifacts={
                    "savedmodel": str(export_path),
                },
                duration_seconds=duration,
            )
            logger.info("Firestore run completed logged.")
        except Exception as exc:
            logger.warning("Firestore completion log failed: %s", exc)

    # ------------------------------------------------------------------
    # W&B finish
    # ------------------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb as _wandb

            _wandb.log(test_metrics)
            _wandb.finish()
        except ImportError:
            pass

    if args.use_wandb:
        try:
            import wandb as _wandb

            _wandb.log(test_metrics)
            _wandb.finish()
        except ImportError:
            pass

    logger.info(
        "Training complete — duration: %.1f min  model: %s",
        duration / 60,
        export_path,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
