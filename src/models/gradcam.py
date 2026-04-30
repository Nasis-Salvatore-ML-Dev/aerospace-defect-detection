"""
Grad-CAM Explainability
src/models/gradcam.py

Purpose: Generate Grad-CAM heatmaps that visually explain which regions
of an input image most influenced the model's defect classification decision.

What is Grad-CAM?
-----------------
Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question:
"Which parts of this image caused the model to predict 'crack'?"

It works by:
1. Running a forward pass and recording the activations of the LAST
   convolutional layer (the final feature maps before GAP): that's what
    we target (high-level, task specific patterns)
2. Computing the gradient of the predicted class score with respect to
   those feature maps via backpropagation.
3. Global-average-pooling the GRADIENTS (not the activations) to get
       one importance weight per feature map channel.
4. Taking a weighted sum of the feature maps using those weights.
5. Applying ReLU to keep only positive contributions.
6. Upsampling the resulting heatmap to the original image size.
7. Overlaying the heatmap on the original image.

Why the last convolutional layer?
----------------------------------
Earlier layers detect low-level features (edges, textures) that are
spatially precise but semantically meaningless.  The last conv layer
detects high-level, task-specific features (crack patterns, surface
anomalies) that directly influence the classification decision.  It
therefore provides the best trade-off between spatial resolution and
semantic meaning.

Why ReLU on the heatmap?
-------------------------
We only care about features that positively contribute to the predicted
class.  Negative activations mean "evidence against this class" — they
are irrelevant for explaining why the model made a specific prediction.
ReLU zeros out negative contributions.

Portfolio value
---------------
Grad-CAM directly satisfies two requirements:
1. "Grad-CAM for visual explainability" (Phase 2 MLOps spec)
2. EU AI Act Article 13 transparency requirement (Phase 3 preview):
   every prediction must be explainable to a human operator.

An aerospace engineer reviewing a "crack" prediction sees exactly where
the model detected the crack — not just a confidence score.

Reference
---------
Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization." ICCV.
https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger("models.gradcam")

# Name of the last convolutional layer in EfficientNetB0.
# This is where Grad-CAM extracts activations and gradients.
# Verify with: [l.name for l in model.get_layer("efficientnetb0").layers]
LAST_CONV_LAYER_NAME = "top_conv"

# Image size must match training
IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------


def compute_gradcam(
    model: keras.Model,
    image_array: np.ndarray,
    class_index: int,
    last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for a specific class.

    Args:
        model:                Trained Keras model (full model including head).
        image_array:          Preprocessed image, shape (1, 224, 224, 3),
                              float32, values in [0, 255].
        class_index:          Index of the class to explain (e.g. 1 for "crack").
        last_conv_layer_name: Name of the last convolutional layer in the backbone.

    Returns:
        heatmap: float32 array, shape (H, W), values in [0, 1].
                 H and W match the spatial dimensions of the last conv layer
                 (typically 7×7 for EfficientNetB0 with 224×224 input).
                 Upsampling to image size happens in overlay_heatmap().

    Step-by-step:
        1. Build a sub-model that outputs both the last conv layer activations
           AND the final softmax predictions simultaneously.
        2. Record gradients of the class score w.r.t. the conv activations
           using tf.GradientTape.
        3. Global-average-pool the gradients → one weight per channel.
        4. Weighted sum of activation channels → raw heatmap.
        5. ReLU → keep only positive contributions.
        6. Normalise to [0, 1].
    """
    # ------------------------------------------------------------------
    # Step 1: Build grad model
    # Outputs: (conv_activations, predictions)
    # We need both simultaneously inside GradientTape.
    # ------------------------------------------------------------------
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    # ------------------------------------------------------------------
    # Step 2: Forward pass inside GradientTape to record gradients
    # ------------------------------------------------------------------
    image_tensor = tf.cast(image_array, tf.float32)

    with tf.GradientTape() as tape:
        # Watch the conv layer output (not a tf.Variable, so must watch explicitly)
        tape.watch(image_tensor)

        # Forward pass — get conv activations and final predictions
        conv_outputs, predictions = grad_model(image_tensor, training=False)

        # Score for the target class (scalar)
        # We use the raw logit-equivalent: softmax output for class_index
        class_score = predictions[:, class_index]

    # ------------------------------------------------------------------
    # Step 3: Compute gradients of class score w.r.t. conv activations
    # shape: (1, H, W, C) where C = number of filters in last conv layer
    # ------------------------------------------------------------------
    grads = tape.gradient(class_score, conv_outputs)

    # ------------------------------------------------------------------
    # Step 4: Global average pool the gradients
    # Mean over spatial dimensions (H, W) → shape: (1, 1, 1, C)
    # Each value is the importance weight for one feature map channel.
    # ------------------------------------------------------------------
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # ------------------------------------------------------------------
    # Step 5: Weighted sum of conv activation channels
    # Multiply each channel by its importance weight, then sum across channels
    # conv_outputs[0]: shape (H, W, C)
    # pooled_grads:    shape (C,)
    # Result:          shape (H, W)
    # ------------------------------------------------------------------
    conv_outputs = conv_outputs[0]  # remove batch dimension → (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)  # (H, W)

    # ------------------------------------------------------------------
    # Step 6: ReLU — keep only positive contributions
    # ------------------------------------------------------------------
    heatmap = tf.nn.relu(heatmap)

    # ------------------------------------------------------------------
    # Step 7: Normalise to [0, 1]
    # ------------------------------------------------------------------
    heatmap = heatmap.numpy()
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        # Edge case: all gradients were zero (model is fully certain)
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Heatmap overlay
# ---------------------------------------------------------------------------


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
    colormap: int = 2,  # cv2.COLORMAP_JET = 2
) -> np.ndarray:
    """
    Upsample the Grad-CAM heatmap to image size and overlay on the original.

    The heatmap is colourised using a jet colourmap (blue=low, red=high)
    and blended with the original image using alpha compositing.

    Args:
        heatmap:        Float32 array, shape (H, W), values in [0, 1].
                        Typically 7×7 from EfficientNetB0.
        original_image: uint8 RGB array, shape (224, 224, 3).
        alpha:          Opacity of the heatmap overlay [0, 1].
                        0.4 means 40% heatmap, 60% original image.
        colormap:       OpenCV colormap integer. Default: COLORMAP_JET (2).

    Returns:
        overlaid: uint8 RGB array, shape (224, 224, 3).
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for heatmap overlay. "
            "Install with: pip install opencv-python-headless"
        ) from exc

    # Upsample heatmap to original image size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Apply colour map (converts grayscale heatmap to BGR colour image)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)

    # Convert original image from RGB to BGR for OpenCV
    original_bgr = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Alpha blend: heatmap * alpha + original * (1 - alpha)
    overlaid_bgr = cv2.addWeighted(heatmap_colored, alpha, original_bgr, 1 - alpha, 0)

    # Convert back to RGB
    overlaid_rgb = cv2.cvtColor(overlaid_bgr, cv2.COLOR_BGR2RGB)

    return overlaid_rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# End-to-end pipeline: image bytes → base64 PNG
# ---------------------------------------------------------------------------


def gradcam_to_base64(
    model: keras.Model,
    image_bytes: bytes,
    class_index: int,
    alpha: float = 0.4,
) -> str:
    """
    Full Grad-CAM pipeline: raw image bytes → base64-encoded PNG.

    This is the function called by the /explain API endpoint.
    It takes the raw image bytes from the client, runs Grad-CAM, overlays
    the heatmap, and returns a base64-encoded PNG ready for display.

    Args:
        model:       Trained Keras model.
        image_bytes: Raw image bytes (JPEG, PNG, etc.).
        class_index: Class index to explain (from the /predict response).
        alpha:       Heatmap overlay opacity.

    Returns:
        base64_png: Base64-encoded string of the overlaid PNG.
                    Render with: <img src="data:image/png;base64,{base64_png}">

    Pipeline:
        1. Decode image bytes → PIL Image → numpy array (224, 224, 3)
        2. Prepare model input: add batch dim → (1, 224, 224, 3)
        3. compute_gradcam() → heatmap (7, 7)
        4. overlay_heatmap() → overlaid RGB image (224, 224, 3)
        5. Encode as PNG → base64 string
    """
    from PIL import Image  # type: ignore

    # ------------------------------------------------------------------
    # Step 1: Decode and resize image
    # ------------------------------------------------------------------
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    original_array = np.array(img_resized, dtype=np.uint8)  # (224, 224, 3)

    # ------------------------------------------------------------------
    # Step 2: Prepare model input
    # EfficientNetB0 with include_preprocessing=True expects float32 [0,255]
    # ------------------------------------------------------------------
    input_array = np.expand_dims(original_array.astype(np.float32), axis=0)

    # ------------------------------------------------------------------
    # Step 3: Compute Grad-CAM heatmap
    # ------------------------------------------------------------------
    heatmap = compute_gradcam(
        model=model,
        image_array=input_array,
        class_index=class_index,
        last_conv_layer_name=LAST_CONV_LAYER_NAME,
    )

    # ------------------------------------------------------------------
    # Step 4: Overlay heatmap on original image
    # ------------------------------------------------------------------
    overlaid = overlay_heatmap(
        heatmap=heatmap,
        original_image=original_array,
        alpha=alpha,
    )

    # ------------------------------------------------------------------
    # Step 5: Encode as PNG → base64
    # ------------------------------------------------------------------
    pil_overlaid = Image.fromarray(overlaid)
    buffer = io.BytesIO()
    pil_overlaid.save(buffer, format="PNG")
    buffer.seek(0)
    base64_png = base64.b64encode(buffer.read()).decode("utf-8")

    logger.info(
        "Grad-CAM generated — class_index=%d  heatmap_shape=%s",
        class_index,
        heatmap.shape,
    )

    return base64_png


# ---------------------------------------------------------------------------
# Standalone script: generate and save a heatmap image
# ---------------------------------------------------------------------------


def save_gradcam_image(
    model: keras.Model,
    image_path: Path,
    class_index: int,
    output_path: Path,
    alpha: float = 0.4,
) -> None:
    """
    Generate a Grad-CAM heatmap for a single image and save to disk.

    Useful for offline analysis, README illustrations, and debugging.

    Args:
        model:       Trained Keras model.
        image_path:  Path to the input image file.
        class_index: Class index to explain.
        output_path: Path to save the overlaid PNG.
        alpha:       Heatmap overlay opacity.
    """

    image_bytes = image_path.read_bytes()
    base64_png = gradcam_to_base64(model, image_bytes, class_index, alpha)

    # Decode base64 back to bytes and save
    png_bytes = base64.b64decode(base64_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)

    logger.info("Grad-CAM image saved to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--class-index", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("gradcam_output.png"))
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = keras.models.load_model(str(args.model_dir))
    save_gradcam_image(
        model=model,
        image_path=args.image,
        class_index=args.class_index,
        output_path=args.output,
        alpha=args.alpha,
    )
    print(f"Saved: {args.output}")
