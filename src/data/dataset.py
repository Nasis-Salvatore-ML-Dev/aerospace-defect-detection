"""tf.data pipeline for MVTec Anomaly Detection dataset.

Reframes MVTec categories as aerospace components:
    metal_nut → fastener
    screw     → structural bolt
    tile      → thermal shield panel
    cable     → wiring harness
    capsule   → pressure vessel component
"""

import logging
from pathlib import Path

import tensorflow as tf

from src.data.augmentation import build_augmentation_pipeline

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

CATEGORY_MAP = {
    "metal_nut": "fastener",
    "screw": "structural_bolt",
    "tile": "thermal_shield",
    "cable": "wiring_harness",
    "capsule": "pressure_vessel",
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_image(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load and preprocess a single image from disk.

    Args:
        path: String tensor containing the file path.
        label: Integer label tensor.
        image_size: Target (height, width) after resizing.

    Returns:
        Tuple of (preprocessed image tensor, label tensor).
    """
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    # Normalize to [0, 1] — EfficientNetB0 expects this range
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_dataset(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    augment: bool = True,
    categories: list[str] | None = None,
) -> tuple[tf.data.Dataset, dict[str, float]]:
    """Build a tf.data.Dataset for the specified split.

    Directory structure expected:
        data_dir/
            {category}/
                train/good/         ← normal images
                test/good/          ← normal test images
                test/{defect_type}/ ← defective images

    Args:
        data_dir: Root directory containing MVTec category folders.
        split: One of 'train', 'val', 'test'.
        batch_size: Number of images per batch.
        image_size: Target (height, width) for resizing.
        augment: Apply augmentation (training only).
        categories: List of categories to include. Defaults to all 5.

    Returns:
        Tuple of (tf.data.Dataset, class_weights dict).
    """
    data_dir = Path(data_dir)
    categories = categories or list(CATEGORY_MAP.keys())

    image_paths: list[str] = []
    labels: list[int] = []  # 0 = normal, 1 = defective

    for category in categories:
        cat_dir = data_dir / category

        if not cat_dir.exists():
            logger.warning("Category directory not found: %s", cat_dir)
            continue

        if split in ("train", "val"):
            # Normal images from training set
            normal_dir = cat_dir / "train" / "good"
            normal_paths = list(normal_dir.glob("*.png")) + list(
                normal_dir.glob("*.jpg")
            )
            image_paths.extend([str(p) for p in normal_paths])
            labels.extend([0] * len(normal_paths))

        if split in ("val", "test"):
            # Normal images from test set
            normal_test_dir = cat_dir / "test" / "good"
            normal_test_paths = list(normal_test_dir.glob("*.png")) + list(
                normal_test_dir.glob("*.jpg")
            )

            # Defective images from test set
            defect_paths: list[str] = []
            test_dir = cat_dir / "test"
            for defect_dir in test_dir.iterdir():
                if defect_dir.name == "good":
                    continue
                defect_paths.extend([str(p) for p in defect_dir.glob("*.png")])
                defect_paths.extend([str(p) for p in defect_dir.glob("*.jpg")])

            image_paths.extend([str(p) for p in normal_test_paths])
            labels.extend([0] * len(normal_test_paths))
            image_paths.extend(defect_paths)
            labels.extend([1] * len(defect_paths))

    if not image_paths:
        raise ValueError(
            f"No images found in {data_dir} for split='{split}'. "
            "Run src/data/download.py first."
        )

    logger.info(
        "Split='%s': %d total images (%d normal, %d defective)",
        split,
        len(labels),
        labels.count(0),
        labels.count(1),
    )

    # ── Compute class weights to handle imbalance ──────────────────────────
    n_normal = labels.count(0)
    n_defective = labels.count(1)
    n_total = len(labels)

    class_weights = {
        0: n_total / (2 * n_normal) if n_normal > 0 else 1.0,
        1: n_total / (2 * n_defective) if n_defective > 0 else 1.0,
    }
    logger.info("Class weights: %s", class_weights)

    # ── Build tf.data pipeline ─────────────────────────────────────────────
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if split == "train":
        path_ds = path_ds.shuffle(
            buffer_size=len(image_paths),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = path_ds.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )

    # Apply augmentation during training only
    if augment and split == "train":
        aug_pipeline = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda image, label: (
                aug_pipeline(image, training=True),
                label,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(batch_size, drop_remainder=(split == "train")).prefetch(
        AUTOTUNE
    )

    if split == "train":
        dataset = dataset.cache()

    return dataset, class_weights
