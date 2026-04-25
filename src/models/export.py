"""
Model Export & Latency Benchmarking
src/models/export.py

Purpose: Export the trained Keras SavedModel to ONNX and TFLite formats,
then benchmark inference latency across all three formats.

Why export?
-----------
SavedModel is the training format — it carries the full Keras graph,
optimizer state, and training config.  For deployment, we need leaner formats:

    ONNX (Open Neural Network Exchange)
        - Runtime-agnostic: runs on ONNX Runtime, TensorRT, OpenVINO, CoreML.
        - Enables deployment outside the TensorFlow ecosystem.
        - Required for edge devices, C++ inference servers, and Windows targets.
        - Typical speedup over SavedModel: 1.5-2.5x on CPU.

    TFLite (TensorFlow Lite)
        - Designed for mobile and edge devices (drones, inspection tablets).
        - INT8 quantization reduces model size ~4x and speeds up inference
          on hardware with integer arithmetic units (ARM Cortex-M, DSPs).
        - Typical speedup over SavedModel: 2-4x on CPU with INT8.

Benchmarking
------------
We measure p50, p95, p99 latency across 100 warm-up + 200 timed runs.
Results are logged to Firestore and printed as a comparison table.
This directly addresses the portfolio requirement: "latency benchmarking
for ONNX / TFLite formats."

Usage
-----
    python src/models/export.py \\
        --savedmodel-dir models/saved_model/v1.0 \\
        --output-dir models/exports \\
        --model-version v1.0 \\
        --quantize \\
        --benchmark \\
        --gcp-project my-gcp-project

Cost: $0 — runs locally or on Colab CPU, no GCP services required.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("export")

# Image dimensions must match training
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SavedModel to ONNX and TFLite, then benchmark."
    )
    parser.add_argument(
        "--savedmodel-dir",
        type=Path,
        required=True,
        help="Path to the trained SavedModel directory, e.g. models/saved_model/v1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/exports"),
        help="Directory to write ONNX and TFLite files.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0",
        help="Semantic version string for Firestore logging.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Apply INT8 post-training quantization to the TFLite export.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run latency benchmark after export.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=200,
        help="Number of timed inference runs for benchmarking.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=100,
        help="Number of warm-up runs before timing starts.",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        help="GCP project ID for Firestore logging. Skipped if None.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dummy input for benchmarking
# ---------------------------------------------------------------------------


def make_dummy_input() -> np.ndarray:
    """
    Generate a random image batch for benchmarking.

    Shape: (1, 224, 224, 3) — single image, RGB, uint8 values [0, 255].
    EfficientNetB0 with include_preprocessing=True expects raw uint8 pixels.
    """
    return np.random.randint(0, 256, size=(1, *IMAGE_SIZE, 3), dtype=np.uint8).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(savedmodel_dir: Path, output_dir: Path, version: str) -> Path:
    """
    Convert Keras SavedModel to ONNX format using tf2onnx.

    tf2onnx converts the TensorFlow graph to an ONNX protobuf.
    The resulting .onnx file can be run with onnxruntime on any platform
    — no TensorFlow installation required at inference time.

    Args:
        savedmodel_dir: Path to the SavedModel directory.
        output_dir:     Directory to write the .onnx file.
        version:        Version string for the filename.

    Returns:
        Path to the exported .onnx file.
    """
    try:
        import tensorflow as tf
        import tf2onnx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tf2onnx is required for ONNX export. " "Install with: pip install tf2onnx"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"aerospace_defect_detector_{version}.onnx"

    logger.info("Loading SavedModel from %s ...", savedmodel_dir)
    model = tf.keras.models.load_model(str(savedmodel_dir))

    logger.info("Converting to ONNX ...")
    input_signature = [
        tf.TensorSpec(
            shape=(None, *IMAGE_SIZE, 3),
            dtype=tf.float32,
            name="image_input",
        )
    ]

    # tf2onnx.convert.from_keras returns (onnx_model, external_tensor_storage)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,  # ONNX opset 13 — widely supported by ONNX Runtime 1.12+
        output_path=str(onnx_path),
    )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    logger.info("ONNX export complete: %s  (%.1f MB)", onnx_path, size_mb)
    return onnx_path


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------


def export_tflite(
    savedmodel_dir: Path,
    output_dir: Path,
    version: str,
    quantize: bool = False,
) -> Path:
    """
    Convert Keras SavedModel to TFLite format.

    Two modes:
        quantize=False  → float32 TFLite model (same precision as SavedModel)
        quantize=True   → INT8 post-training quantization

    INT8 quantization
    -----------------
    Converts weights and activations from float32 to int8.
    - Model size: ~4x smaller (from ~29MB to ~7MB for EfficientNetB0)
    - Latency:    2-4x faster on ARM CPUs with NEON integer units
    - Accuracy:   typically <1% degradation on val_recall

    The quantization uses a representative dataset of 100 random images
    to calibrate the int8 scale factors for each layer.  In production,
    use real validation images for more accurate calibration.

    Args:
        savedmodel_dir: Path to the SavedModel directory.
        output_dir:     Directory to write the .tflite file.
        version:        Version string for the filename.
        quantize:       If True, apply INT8 post-training quantization.

    Returns:
        Path to the exported .tflite file.
    """
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_int8" if quantize else "_fp32"
    tflite_path = output_dir / f"aerospace_defect_detector_{version}{suffix}.tflite"

    logger.info("Loading SavedModel from %s ...", savedmodel_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))

    if quantize:
        logger.info("Applying INT8 post-training quantization ...")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        def representative_dataset():
            """
            Yield 100 random image batches for INT8 calibration.

            In production, replace with real validation images:
                for img_path in val_image_paths[:100]:
                    img = load_and_preprocess(img_path)
                    yield [img]
            """
            for _ in range(100):
                yield [make_dummy_input()]

        converter.representative_dataset = representative_dataset
    else:
        logger.info("Exporting float32 TFLite model ...")

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = tflite_path.stat().st_size / 1024 / 1024
    logger.info("TFLite export complete: %s  (%.1f MB)", tflite_path, size_mb)
    return tflite_path


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------


def benchmark_savedmodel(
    savedmodel_dir: Path,
    n_runs: int,
    warmup: int,
) -> dict[str, float]:
    """
    Benchmark SavedModel inference latency.

    Returns:
        Dict with p50, p95, p99 latency in milliseconds.
    """
    import tensorflow as tf

    logger.info("Benchmarking SavedModel (%d runs + %d warmup) ...", n_runs, warmup)
    model = tf.keras.models.load_model(str(savedmodel_dir))
    dummy = make_dummy_input()

    # Warm-up
    for _ in range(warmup):
        model.predict(dummy, verbose=0)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)

    return _latency_stats(latencies, "SavedModel")


def benchmark_onnx(onnx_path: Path, n_runs: int, warmup: int) -> dict[str, float]:
    """
    Benchmark ONNX Runtime inference latency.

    Returns:
        Dict with p50, p95, p99 latency in milliseconds.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX benchmarking. "
            "Install with: pip install onnxruntime"
        ) from exc

    logger.info("Benchmarking ONNX (%d runs + %d warmup) ...", n_runs, warmup)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    dummy = make_dummy_input()

    # Warm-up
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    return _latency_stats(latencies, "ONNX")


def benchmark_tflite(tflite_path: Path, n_runs: int, warmup: int) -> dict[str, float]:
    """
    Benchmark TFLite interpreter inference latency.

    Returns:
        Dict with p50, p95, p99 latency in milliseconds.
    """
    import tensorflow as tf

    logger.info("Benchmarking TFLite (%d runs + %d warmup) ...", n_runs, warmup)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    dummy = make_dummy_input()

    # Warm-up
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    return _latency_stats(latencies, "TFLite")


def _latency_stats(latencies: list[float], label: str) -> dict[str, float]:
    """Compute and log p50/p95/p99 from a list of latency measurements."""
    arr = np.array(latencies)
    stats = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }
    logger.info(
        "%s latency — p50: %.1f ms  p95: %.1f ms  p99: %.1f ms  mean: %.1f ms",
        label,
        stats["p50_ms"],
        stats["p95_ms"],
        stats["p99_ms"],
        stats["mean_ms"],
    )
    return stats


def print_benchmark_table(results: dict[str, dict[str, float]]) -> None:
    """
    Print a formatted comparison table of latency results.

    Example output:
        Format        p50 (ms)    p95 (ms)    p99 (ms)    Speedup vs SavedModel
        SavedModel      85.3        92.1        98.4        1.0x
        ONNX            42.7        48.3        51.2        2.0x
        TFLite INT8     24.1        27.8        30.2        3.5x
    """
    baseline = results.get("SavedModel", {}).get("p50_ms", 1.0)

    header = (
        f"{'Format':<20} {'p50 (ms)':>10} {'p95 (ms)':>10}"
        f" {'p99 (ms)':>10} {'Speedup':>10}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for fmt, stats in results.items():
        speedup = baseline / stats["p50_ms"] if stats["p50_ms"] > 0 else 0.0
        print(
            f"{fmt:<20} "
            f"{stats['p50_ms']:>10.1f} "
            f"{stats['p95_ms']:>10.1f} "
            f"{stats['p99_ms']:>10.1f} "
            f"{speedup:>9.1f}x"
        )

    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run export and optional benchmarking pipeline."""

    if not args.savedmodel_dir.exists():
        raise FileNotFoundError(
            f"SavedModel directory not found: {args.savedmodel_dir}\n"
            f"Run training first: python src/models/train.py"
        )

    benchmark_results: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    onnx_path = export_onnx(
        savedmodel_dir=args.savedmodel_dir,
        output_dir=args.output_dir,
        version=args.model_version,
    )

    # ------------------------------------------------------------------
    # TFLite export (float32)
    # ------------------------------------------------------------------
    tflite_fp32_path = export_tflite(
        savedmodel_dir=args.savedmodel_dir,
        output_dir=args.output_dir,
        version=args.model_version,
        quantize=False,
    )

    # ------------------------------------------------------------------
    # TFLite export (INT8 quantized)
    # ------------------------------------------------------------------
    tflite_int8_path = None
    if args.quantize:
        tflite_int8_path = export_tflite(
            savedmodel_dir=args.savedmodel_dir,
            output_dir=args.output_dir,
            version=args.model_version,
            quantize=True,
        )

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    if args.benchmark:
        benchmark_results["SavedModel"] = benchmark_savedmodel(
            args.savedmodel_dir, args.benchmark_runs, args.warmup_runs
        )
        benchmark_results["ONNX"] = benchmark_onnx(
            onnx_path, args.benchmark_runs, args.warmup_runs
        )
        benchmark_results["TFLite FP32"] = benchmark_tflite(
            tflite_fp32_path, args.benchmark_runs, args.warmup_runs
        )
        if tflite_int8_path:
            benchmark_results["TFLite INT8"] = benchmark_tflite(
                tflite_int8_path, args.benchmark_runs, args.warmup_runs
            )

        print_benchmark_table(benchmark_results)

    # ------------------------------------------------------------------
    # Firestore: log export artifacts and benchmark results
    # ------------------------------------------------------------------
    if args.gcp_project and benchmark_results:
        try:
            from src.monitoring.firestore_logger import FirestoreLogger

            fs = FirestoreLogger(project_id=args.gcp_project)
            best_run = fs.get_best_run(metric="val_recall")

            if best_run:
                artifacts = {
                    "onnx": str(onnx_path),
                    "tflite_fp32": str(tflite_fp32_path),
                }
                if tflite_int8_path:
                    artifacts["tflite_int8"] = str(tflite_int8_path)

                fs._update(
                    "training_runs",
                    best_run["run_id"],
                    {
                        "artifacts": artifacts,
                        "benchmark": benchmark_results,
                    },
                )
                logger.info("Firestore updated with export artifacts and benchmark.")
        except Exception as exc:
            logger.warning("Firestore update failed (continuing): %s", exc)

    logger.info("Export pipeline complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
