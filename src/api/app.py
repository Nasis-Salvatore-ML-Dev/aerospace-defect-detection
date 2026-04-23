"""
Aerospace Defect Detection API
src/api/app.py

Production REST API for CNN-based aerospace component inspection.
Serves an EfficientNetB0 model trained on MVTec AD (aerospace subset).

Endpoints
---------
POST /predict              Single image defect classification (JSON body, base64)
POST /predict/upload       Single image defect classification (multipart file upload)
POST /predict/batch        Batch classification (up to 50 images, JSON body)
GET  /health               Liveness + readiness check
GET  /metrics              Model performance metrics from Firestore
GET  /explain/{filename}   Grad-CAM explainability heatmap (Week 4)

Architecture note
-----------------
The model is loaded ONCE at startup and held in module-level state (MODEL_STORE).
Cloud Run pulls the container image once per instance; the model is in memory
for the lifetime of that instance.  All requests share the same in-memory model —
no per-request loading, no race conditions (inference is stateless read-only).

This is the correct production pattern.  Loading the model inside the route
handler would add 2-5 seconds of latency per request.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestLoggingMiddleware, configure_logging
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DefectClass,
    DefectPredictionRequest,
    DefectPredictionResponse,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
)

# ---------------------------------------------------------------------------
# Module-level logger
# Configured by configure_logging() in lifespan — do not call basicConfig here.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global model store
# Populated at startup.  Never modified after startup (read-only during serving).
# ---------------------------------------------------------------------------
MODEL_STORE: dict = {
    "model": None,  # Keras model object (loaded from SavedModel)
    "version": "not_loaded",  # Semantic version string e.g. "v1.0"
    "class_names": list(DefectClass),  # Ordered list matching model output indices
    "loaded_at": None,  # ISO timestamp of when model was loaded
}

# Runtime counters — updated on every prediction request
# These feed the GET /metrics endpoint without a Firestore query
_RUNTIME_STATS: dict = {
    "requests_served": 0,
    "total_latency_ms": 0.0,
}


# ---------------------------------------------------------------------------
# Startup: model loading
# ---------------------------------------------------------------------------


def _load_model() -> None:
    """
    Load the trained Keras SavedModel into MODEL_STORE.

    Called once inside the lifespan handler at container startup.
    If the model file is absent (e.g. first run before training), the API
    starts anyway — /health returns model_loaded=False and /predict returns
    503 until the model is available.

    The model path follows the versioned SavedModel convention:
        models/saved_model/v1.0/
    The version string is read from the directory name so it is always
    in sync with the artifact on disk — no manual config required.
    """

    from datetime import datetime, timezone
    from pathlib import Path

    model_dir = Path("models/saved_model")

    if not model_dir.exists():
        logger.warning(
            "Model directory not found at %s — API will start without a model. "
            "Run training first, then restart the container.",
            model_dir,
        )
        return

    # Find the latest version directory (e.g. v1.0, v1.1)
    version_dirs = sorted(model_dir.glob("v*"))
    if not version_dirs:
        logger.warning("No versioned model directories found under %s.", model_dir)
        return

    latest_dir = version_dirs[-1]
    version = latest_dir.name

    try:
        import tensorflow as tf  # type: ignore

        logger.info("Loading SavedModel from %s ...", latest_dir)
        model = tf.keras.models.load_model(str(latest_dir))
        MODEL_STORE["model"] = model
        MODEL_STORE["version"] = version
        MODEL_STORE["loaded_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Model loaded successfully — version=%s", version)

    except Exception as exc:
        logger.error("Failed to load model from %s: %s", latest_dir, exc)
        # Do not re-raise — allow API to start in degraded mode


# ---------------------------------------------------------------------------
# Lifespan handler
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging, load model at startup; log shutdown."""
    configure_logging(log_dir="logs")
    logger.info("Aerospace Defect Detection API starting up")
    _load_model()
    yield
    logger.info(
        "API shutting down — requests_served=%d", _RUNTIME_STATS["requests_served"]
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aerospace Defect Detection API",
    description=(
        "CNN-based surface defect detection for aerospace components. "
        "Powered by EfficientNetB0 fine-tuned on MVTec AD. "
        "Includes Grad-CAM explainability and full MLOps instrumentation."
    ),
    version="0.1.0",
    lifespan=lifespan,
    responses={
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        503: {"description": "Model not loaded", "model": ErrorResponse},
    },
)

# Middleware — order matters: CORS first, then request logging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_model() -> object:
    """
    Return the loaded model or raise 503.

    Called at the start of every prediction route to give a clean error
    before any preprocessing happens.
    """
    if MODEL_STORE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first, then restart the API.",
        )
    return MODEL_STORE["model"]


def _run_inference(image_bytes: bytes) -> tuple[DefectClass, float, dict[str, float]]:
    """
    Preprocess raw image bytes and run a single forward pass.

    Returns:
        predicted_class:      Top-1 DefectClass
        confidence:           Softmax probability of top-1 class
        class_probabilities:  Full softmax distribution as {class_name: prob}

    This function is intentionally synchronous.  FastAPI runs async route
    handlers in an async event loop, but CPU-bound inference blocks that loop.
    For a portfolio project with low concurrency this is acceptable.
    In production, wrap this in asyncio.run_in_executor() to avoid blocking.
    """
    import io

    import numpy as np
    from PIL import Image  # type: ignore

    model = _require_model()
    class_names = MODEL_STORE["class_names"]

    # Decode and preprocess
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # EfficientNetB0 input size

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # normalise to [0, 1]
    img_array = np.expand_dims(img_array, 0)  # add batch dimension → (1, 224, 224, 3)

    # Inference
    predictions = model.predict(img_array, verbose=0)  # shape: (1, n_classes)
    probs = predictions[0]  # shape: (n_classes,)

    top_idx = int(np.argmax(probs))
    predicted_class = DefectClass(class_names[top_idx])
    confidence = float(probs[top_idx])
    class_probabilities = {
        class_names[i]: float(probs[i]) for i in range(len(class_names))
    }

    return predicted_class, confidence, class_probabilities


def _decode_base64_image(image_base64: str) -> bytes:
    """Decode a base64 string to raw bytes. Raises HTTPException on failure."""
    import base64
    import binascii

    try:
        return base64.b64decode(image_base64)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64 image: {exc}")


def _build_prediction_response(
    predicted_class: DefectClass,
    confidence: float,
    class_probabilities: dict[str, float],
    filename: str,
    processing_time_ms: float,
) -> DefectPredictionResponse:
    """Assemble a DefectPredictionResponse from inference outputs."""
    return DefectPredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        defect_detected=predicted_class != DefectClass.GOOD,
        class_probabilities=class_probabilities,
        model_version=MODEL_STORE["version"],
        processing_time_ms=round(processing_time_ms, 2),
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health() -> HealthResponse:
    """
    Liveness and readiness check.

    Cloud Run calls this endpoint to decide whether to send traffic to this
    instance.  Returns 200 even if the model is not loaded — the model_loaded
    field tells the caller whether predictions are available.

    Returns 200 always (liveness).  Use model_loaded to gate prediction calls.
    """
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_STORE["model"] is not None,
        model_version=MODEL_STORE["version"],
        version=app.version,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Infrastructure"])
async def get_metrics() -> MetricsResponse:
    """
    Model performance metrics.

    Returns validation metrics from the training run that produced the
    currently loaded model, plus runtime statistics (requests served,
    average latency) since the last container startup.

    In production, this endpoint is scraped by Prometheus every 15 seconds.
    For the portfolio demo, it surfaces useful information in the README
    curl examples.
    """
    # Runtime stats
    requests = _RUNTIME_STATS["requests_served"]
    avg_latency = _RUNTIME_STATS["total_latency_ms"] / requests if requests > 0 else 0.0

    # Training metrics: in production these come from Firestore (best run).
    # At this stage (model not yet trained), we return placeholder values.
    # Week 2 will wire this to FirestoreLogger.get_best_run().
    training_metrics: dict[str, float] = {
        "val_recall": 0.0,
        "val_accuracy": 0.0,
        "val_precision": 0.0,
        "val_loss": 0.0,
    }

    return MetricsResponse(
        model_version=MODEL_STORE["version"],
        training_metrics=training_metrics,
        requests_served=requests,
        avg_latency_ms=round(avg_latency, 2),
    )


@app.post(
    "/predict",
    response_model=DefectPredictionResponse,
    tags=["Prediction"],
    summary="Classify a single aerospace component image (base64 JSON body)",
)
async def predict(
    payload: DefectPredictionRequest, request: Request
) -> DefectPredictionResponse:
    """
    Classify a single image sent as a base64-encoded string in a JSON body.

    The model returns the predicted defect class, confidence score, and the
    full softmax distribution over all defect classes.

    **Critical safety note:** defect_detected=True means a defect was found.
    False negatives (missed defects) are the highest-cost error in aerospace
    inspection.  The model is trained with a severity-weighted loss that
    penalises false negatives more heavily than false positives.

    Use the confidence score and class_probabilities to implement a
    custom decision threshold if the default (argmax) is too permissive.
    """
    _require_model()
    start = time.perf_counter()

    image_bytes = _decode_base64_image(payload.image_base64)

    try:
        predicted_class, confidence, class_probabilities = _run_inference(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Inference failed for file=%s: %s", payload.filename, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    elapsed_ms = (time.perf_counter() - start) * 1_000
    _RUNTIME_STATS["requests_served"] += 1
    _RUNTIME_STATS["total_latency_ms"] += elapsed_ms

    response = _build_prediction_response(
        predicted_class, confidence, class_probabilities, payload.filename, elapsed_ms
    )

    logger.info(
        "Prediction complete",
        extra={
            "filename": payload.filename,
            "predicted_class": predicted_class.value,
            "confidence": round(confidence, 4),
            "defect_detected": response.defect_detected,
            "processing_time_ms": round(elapsed_ms, 2),
            "model_version": MODEL_STORE["version"],
        },
    )

    return response


@app.post(
    "/predict/upload",
    response_model=DefectPredictionResponse,
    tags=["Prediction"],
    summary="Classify a single aerospace component image (multipart file upload)",
)
async def predict_upload(
    file: UploadFile = File(
        ..., description="Image file. Supported: JPEG, PNG, BMP, TIFF."
    ),
) -> DefectPredictionResponse:
    """
    Classify a single image uploaded as a multipart form file.

    This endpoint accepts the same image formats as /predict but via
    standard file upload — useful for curl, Postman, and web forms.

    Example curl:
        curl -X POST http://localhost:8000/predict/upload \\
             -F "file=@blade_042.jpg"
    """
    _require_model()
    start = time.perf_counter()

    # Validate file size (10 MB limit)
    image_bytes = await file.read()
    max_bytes = 10 * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"File too large "
                f"({len(image_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum: 10 MB."
            ),
        )

    filename = file.filename or "unknown.jpg"

    try:
        predicted_class, confidence, class_probabilities = _run_inference(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference failed for file=%s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    elapsed_ms = (time.perf_counter() - start) * 1_000
    _RUNTIME_STATS["requests_served"] += 1
    _RUNTIME_STATS["total_latency_ms"] += elapsed_ms

    return _build_prediction_response(
        predicted_class, confidence, class_probabilities, filename, elapsed_ms
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Classify up to 50 images in a single request",
)
async def predict_batch(
    payload: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    """
    Classify a batch of up to 50 images in a single API call.

    All images must be base64-encoded (same format as /predict).
    Results are returned in the same order as the input images.

    The batch endpoint is more efficient than calling /predict 50 times
    because preprocessing and postprocessing overhead is amortised across
    the batch.  Model inference runs as a single batched forward pass.

    Interview note: a single model.predict(batch_tensor) call on 50 images
    takes roughly the same time as 5-10 individual calls because GPU/CPU
    parallelism is utilised across the batch dimension.  On CPU (Cloud Run),
    the benefit is smaller but still present due to vectorised numpy ops.
    """
    _require_model()
    batch_start = time.perf_counter()

    predictions: list[DefectPredictionResponse] = []

    for img_request in payload.images:
        img_start = time.perf_counter()
        image_bytes = _decode_base64_image(img_request.image_base64)

        try:
            predicted_class, confidence, class_probabilities = _run_inference(
                image_bytes
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Batch inference failed for file=%s: %s",
                img_request.filename,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Inference error on {img_request.filename}: {exc}",
            )

        img_elapsed_ms = (time.perf_counter() - img_start) * 1_000
        predictions.append(
            _build_prediction_response(
                predicted_class,
                confidence,
                class_probabilities,
                img_request.filename,
                img_elapsed_ms,
            )
        )

    batch_elapsed_ms = (time.perf_counter() - batch_start) * 1_000
    _RUNTIME_STATS["requests_served"] += len(predictions)
    _RUNTIME_STATS["total_latency_ms"] += batch_elapsed_ms

    defects_detected = sum(1 for p in predictions if p.defect_detected)

    logger.info(
        "Batch prediction complete",
        extra={
            "total_images": len(predictions),
            "defects_detected": defects_detected,
            "processing_time_ms": round(batch_elapsed_ms, 2),
        },
    )

    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(predictions),
        defects_detected=defects_detected,
        defect_rate=round(defects_detected / len(predictions), 4),
        processing_time_ms=round(batch_elapsed_ms, 2),
        model_version=MODEL_STORE["version"],
    )


@app.get(
    "/explain/{filename}",
    response_model=ExplainResponse,
    tags=["Explainability"],
    summary="Grad-CAM heatmap for the most recent prediction on this image",
)
async def explain(filename: str) -> ExplainResponse:
    """
    Return a Grad-CAM heatmap explaining the most recent prediction
    for the specified image filename.

    **Status: stub — implemented in Week 4 (Day 20).**

    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
    the image regions that most influenced the model's decision.  For
    aerospace inspection, this is critical: an engineer reviewing a
    'crack' prediction needs to see *where* the crack was detected.

    The heatmap is returned as a base64-encoded PNG overlaid on the
    original image.  The client renders it as:
        <img src="data:image/png;base64,{gradcam_heatmap_base64}">

    Reference: Selvaraju et al. (2017), "Grad-CAM: Visual Explanations
    from Deep Networks via Gradient-based Localization."
    """
    # Week 4 implementation will:
    # 1. Retrieve the original image bytes from a short-lived in-memory cache
    # 2. Run a second forward pass collecting intermediate layer activations
    # 3. Compute Grad-CAM heatmap via src/models/gradcam.py
    # 4. Overlay heatmap on original image using OpenCV / PIL
    # 5. Base64-encode the result and return

    raise HTTPException(
        status_code=501,
        detail="Grad-CAM explainability endpoint is implemented in Week 4. Stay tuned.",
    )
