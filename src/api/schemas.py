"""
API Request and Response Schemas
src/api/schemas.py

Purpose: Define all Pydantic models that govern what the API accepts and
returns.  Pydantic enforces these contracts automatically — if a client sends
a malformed request, FastAPI returns a 422 with a precise error message before
any business logic runs.

Design decisions
----------------
1. Two input strategies supported:
   - Base64-encoded image string  (JSON body — easy for programmatic clients)
   - File upload via multipart    (handled in app.py via FastAPI's UploadFile —
                                   not a Pydantic schema but documented here)

2. Defect classes match the MVTec AD dataset subset used for training.
   Adding a new class requires updating VALID_DEFECT_CLASSES and retraining —
   the schema acts as a single source of truth for the output vocabulary.

3. All response models include model_version and processing_time_ms.
   This is intentional: every response must be traceable to the model that
   produced it, and latency must be visible for benchmarking across
   SavedModel / ONNX / TFLite formats.

4. GradCAM heatmap is returned as a base64-encoded PNG in the /explain
   endpoint response.  This keeps the API stateless — no file system writes,
   no GCS uploads required for a single explanation request.
"""

from __future__ import annotations

import base64
import binascii
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Defect class vocabulary
# ---------------------------------------------------------------------------


class DefectClass(str, Enum):
    """
    Output vocabulary for the defect classifier.

    These labels map directly to the MVTec AD categories used during training.
    'good' means no defect detected.  All other values indicate a specific
    defect type found on the aerospace component.

    Interview note: this Enum is the single source of truth for the output
    vocabulary.  If a new defect type is added, it is added here and in the
    training pipeline — nowhere else.
    """

    GOOD = "good"  # No defect — component passes inspection
    CRACK = "crack"  # Surface crack
    SCRATCH = "scratch"  # Surface scratch
    BENT = "bent"  # Structural deformation
    COLOR = "color"  # Colour anomaly / coating failure
    CONTAMINATION = "contamination"  # Foreign material on surface
    HOLE = "hole"  # Missing material / perforation
    BROKEN = "broken"  # Structural break


# Convenience set used by validators
DEFECT_CLASSES = {cls.value for cls in DefectClass}

# Maximum image size we accept: 10 MB as base64 string
# Base64 overhead is ~33%, so 10 MB base64 ≈ 7.5 MB raw image
_MAX_BASE64_BYTES = 10 * 1024 * 1024

# Supported image formats (checked via magic bytes after decode)
_SUPPORTED_FORMATS = {"jpeg", "jpg", "png", "bmp", "tiff"}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class DefectPredictionRequest(BaseModel):
    """
    Example request body:
    {
        "image_base64": "<base64-encoded PNG or JPEG>",
        "filename": "blade_0042.jpg",
        "model_version": "v1.0"
    }
    """

    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded image. Supported formats: JPEG, PNG, BMP, TIFF. "
            "Maximum decoded size: 10 MB."
        ),
        min_length=4,  # shortest valid base64 string
    )

    filename: str = Field(
        default="unknown.jpg",
        description=(
            "Original filename. Used for logging and Firestore prediction records."
        ),
        max_length=255,
    )

    model_version: Optional[str] = Field(
        default=None,
        description=(
            "Request a specific model version, e.g. 'v1.0'. "
            "If None, the currently loaded model is used."
        ),
        pattern=r"^v\d+\.\d+$",  # must match v<major>.<minor>
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """
        Verify the string is valid base64 and within size limits.

        We do NOT decode the full image here — that happens in the route
        handler where we have access to PIL/TF.  We only verify that the
        string is decodable and within our size limit.
        """
        # Strip whitespace that some clients add (e.g. line breaks every 76 chars)
        v = v.strip().replace("\n", "").replace("\r", "")

        # Size check on the base64 string itself
        if len(v.encode()) > _MAX_BASE64_BYTES:
            raise ValueError(
                f"Image exceeds maximum size of 10 MB. "
                f"Received: {len(v.encode()) / 1024 / 1024:.1f} MB."
            )

        # Verify it is valid base64
        try:
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(
                "image_base64 is not valid base64. "
                "Encode your image with: base64.b64encode(image_bytes).decode('utf-8')"
            )

        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Reject path traversal attempts and unsupported extensions."""
        # Reject path separators — only bare filenames accepted
        if "/" in v or "\\" in v:
            raise ValueError("filename must not contain path separators.")

        # Check extension
        suffix = v.rsplit(".", 1)[-1].lower() if "." in v else ""
        if suffix not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file extension '.{suffix}'. "
                f"Supported: {sorted(_SUPPORTED_FORMATS)}"
            )

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_base64": "<base64-encoded JPEG>",
                "filename": "turbine_blade_042.jpg",
                "model_version": None,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Input schema for POST /predict/batch.

    Accepts up to 50 images per request.  Each image is a
    DefectPredictionRequest — the same validation rules apply per image.

    Design note: batch size is capped at 50 because EfficientNetB0 inference
    on CPU (Cloud Run without GPU) at 224×224 takes ~30ms per image.
    50 images × 30ms = ~1.5s total, which stays within Cloud Run's default
    60s request timeout with headroom for preprocessing and postprocessing.
    """

    images: list[DefectPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of images to classify. Minimum 1, maximum 50.",
    )

    @model_validator(mode="after")
    def check_unique_filenames(self) -> "BatchPredictionRequest":
        """
        Warn if duplicate filenames are present.

        Duplicates are not rejected (the client may legitimately send the same
        image twice) but we flag it so the Firestore log is unambiguous.
        """
        filenames = [img.filename for img in self.images]
        if len(filenames) != len(set(filenames)):
            # We cannot raise here without rejecting valid requests.
            # The route handler will log a warning.
            pass
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "images": [
                    {
                        "image_base64": "<base64-encoded JPEG>",
                        "filename": "blade_001.jpg",
                        "model_version": None,
                    },
                    {
                        "image_base64": "<base64-encoded JPEG>",
                        "filename": "blade_002.jpg",
                        "model_version": None,
                    },
                ]
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DefectPredictionResponse(BaseModel):
    """
    Output schema for POST /predict (single image).

    Fields are ordered from most to least important for a quick scan.

    confidence: softmax probability of the predicted class.
    defect_detected: convenience boolean — True for any class except 'good'.
    class_probabilities: full softmax distribution (all classes).
      Included because:
        a) Enables threshold tuning: a caller may want to flag predictions
           where the top-2 classes are close (ambiguous).
        b) Required for drift detection: we monitor the distribution of
           predicted probabilities over time, not just the top-1 class.
    """

    predicted_class: DefectClass = Field(
        ...,
        description="Top-1 predicted defect class.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted class [0, 1].",
    )

    defect_detected: bool = Field(
        ...,
        description="True if predicted_class != 'good'.",
    )

    class_probabilities: dict[str, float] = Field(
        ...,
        description=(
            "Full softmax distribution over all defect classes. "
            "Keys are DefectClass values; values sum to 1.0."
        ),
    )

    model_version: str = Field(
        ...,
        description="Version of the model that produced this prediction.",
    )

    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end inference latency in milliseconds.",
    )

    filename: str = Field(
        ...,
        description="Filename of the input image (echoed from request).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": "crack",
                "confidence": 0.94,
                "defect_detected": True,
                "class_probabilities": {
                    "good": 0.02,
                    "crack": 0.94,
                    "scratch": 0.02,
                    "bent": 0.01,
                    "color": 0.00,
                    "contamination": 0.00,
                    "hole": 0.00,
                    "broken": 0.01,
                },
                "model_version": "v1.0",
                "processing_time_ms": 28.4,
                "filename": "turbine_blade_042.jpg",
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """
    Output schema for POST /predict/batch.

    summary provides aggregate statistics useful for quick quality checks
    without iterating over all individual predictions.
    """

    predictions: list[DefectPredictionResponse] = Field(
        ...,
        description="Prediction result for each input image, in request order.",
    )

    total_images: int = Field(
        ...,
        ge=1,
        description="Number of images processed.",
    )

    defects_detected: int = Field(
        ...,
        ge=0,
        description="Number of images where a defect was detected.",
    )

    defect_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of images with a detected defect.",
    )

    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total wall-clock time for the entire batch in milliseconds.",
    )

    model_version: str = Field(
        ...,
        description="Model version used for all predictions in this batch.",
    )


class HealthResponse(BaseModel):
    """Output schema for GET /health."""

    status: str = Field(..., description="'ok' if the service is healthy.")
    model_loaded: bool = Field(..., description="True if the model is in memory.")
    model_version: str = Field(..., description="Version of the loaded model.")
    version: str = Field(..., description="API version string.")


class MetricsResponse(BaseModel):
    """
    Output schema for GET /metrics.

    Returns the performance metrics from the most recent training run
    stored in Firestore, plus runtime statistics from the current process.

    Interview note: exposing model metrics via API is an MLOps pattern —
    it allows monitoring tools to scrape performance without file system access.
    """

    model_version: str = Field(
        ..., description="Version of the currently loaded model."
    )
    training_metrics: dict[str, float] = Field(
        ...,
        description=(
            "Validation metrics from the training run that produced this model. "
            "Keys: val_recall, val_accuracy, val_precision, val_loss."
        ),
    )
    requests_served: int = Field(
        ...,
        ge=0,
        description="Total prediction requests served since last startup.",
    )
    avg_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Average inference latency in milliseconds since last startup.",
    )


class ExplainResponse(BaseModel):
    """
    Output schema for GET /explain/{filename}.

    Returns the Grad-CAM heatmap for the most recent prediction on
    the specified image, overlaid on the original image.

    The heatmap is base64-encoded PNG — no file system writes needed.
    The client can render it directly: <img src="data:image/png;base64,...">
    """

    filename: str = Field(..., description="Filename of the explained image.")
    predicted_class: DefectClass = Field(..., description="Prediction being explained.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    gradcam_heatmap_base64: str = Field(
        ...,
        description=(
            "Base64-encoded PNG of the Grad-CAM heatmap overlaid on the input image. "
            "Highlights the image regions that most influenced the prediction."
        ),
    )
    model_version: str = Field(..., description="Model version used.")


# ---------------------------------------------------------------------------
# Error schema
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """
    Standardised error response body.

    FastAPI returns this shape for all 4xx and 5xx errors raised by our code.
    Using a consistent error schema makes client error handling predictable.
    """

    error: str = Field(
        ..., description="Short error code, e.g. 'INVALID_IMAGE_FORMAT'."
    )
    detail: str = Field(
        ..., description="Human-readable explanation of what went wrong."
    )
    request_id: Optional[str] = Field(
        default=None,
        description="X-Request-ID from the request header, for log correlation.",
    )
