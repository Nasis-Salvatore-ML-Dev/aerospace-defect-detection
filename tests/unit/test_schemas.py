"""Unit tests for src/api/schemas.py."""

import base64

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchPredictionRequest,
    DefectClass,
    DefectPredictionRequest,
    DefectPredictionResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_valid_base64_image() -> str:
    """Generate a minimal valid base64-encoded JPEG-like bytes string."""
    raw = b"\xff\xd8\xff" + b"\x00" * 100  # JPEG magic bytes + padding
    return base64.b64encode(raw).decode("utf-8")


VALID_B64 = make_valid_base64_image()


# ---------------------------------------------------------------------------
# DefectClass
# ---------------------------------------------------------------------------


class TestDefectClass:
    def test_all_classes_present(self):
        values = {cls.value for cls in DefectClass}
        assert "good" in values
        assert "crack" in values
        assert "scratch" in values
        assert len(values) == 8

    def test_string_enum(self):
        assert DefectClass.GOOD == "good"
        assert DefectClass.CRACK == "crack"


# ---------------------------------------------------------------------------
# DefectPredictionRequest
# ---------------------------------------------------------------------------


class TestDefectPredictionRequest:
    def test_valid_request(self):
        req = DefectPredictionRequest(
            image_base64=VALID_B64,
            filename="blade_042.jpg",
        )
        assert req.filename == "blade_042.jpg"
        assert req.model_version is None

    def test_default_filename(self):
        req = DefectPredictionRequest(image_base64=VALID_B64)
        assert req.filename == "unknown.jpg"

    def test_invalid_base64_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            DefectPredictionRequest(
                image_base64="not-valid-base64!!!",
                filename="test.jpg",
            )
        assert "base64" in str(exc_info.value).lower()

    def test_path_traversal_rejected(self):
        with pytest.raises(ValidationError):
            DefectPredictionRequest(
                image_base64=VALID_B64,
                filename="../etc/passwd",
            )

    def test_valid_model_version(self):
        req = DefectPredictionRequest(
            image_base64=VALID_B64,
            filename="test.jpg",
            model_version="v1.0",
        )
        assert req.model_version == "v1.0"

    def test_invalid_model_version_rejected(self):
        with pytest.raises(ValidationError):
            DefectPredictionRequest(
                image_base64=VALID_B64,
                filename="test.jpg",
                model_version="version1",
            )

    def test_unsupported_extension_rejected(self):
        with pytest.raises(ValidationError):
            DefectPredictionRequest(
                image_base64=VALID_B64,
                filename="test.pdf",
            )


# ---------------------------------------------------------------------------
# BatchPredictionRequest
# ---------------------------------------------------------------------------


class TestBatchPredictionRequest:
    def _make_image_request(self, filename: str) -> dict:
        return {"image_base64": VALID_B64, "filename": filename}

    def test_valid_batch(self):
        req = BatchPredictionRequest(
            images=[
                self._make_image_request("blade_001.jpg"),
                self._make_image_request("blade_002.jpg"),
            ]
        )
        assert len(req.images) == 2

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(images=[])

    def test_batch_too_large_rejected(self):
        images = [self._make_image_request(f"blade_{i:03d}.jpg") for i in range(51)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(images=images)

    def test_max_batch_accepted(self):
        images = [self._make_image_request(f"blade_{i:03d}.jpg") for i in range(50)]
        req = BatchPredictionRequest(images=images)
        assert len(req.images) == 50


# ---------------------------------------------------------------------------
# DefectPredictionResponse
# ---------------------------------------------------------------------------


class TestDefectPredictionResponse:
    def _make_response(self, **kwargs) -> DefectPredictionResponse:
        defaults = {
            "predicted_class": DefectClass.CRACK,
            "confidence": 0.94,
            "defect_detected": True,
            "class_probabilities": {cls.value: 0.0 for cls in DefectClass},
            "model_version": "v1.0",
            "processing_time_ms": 28.4,
            "filename": "blade_042.jpg",
        }
        defaults.update(kwargs)
        return DefectPredictionResponse(**defaults)

    def test_valid_response(self):
        resp = self._make_response()
        assert resp.predicted_class == DefectClass.CRACK
        assert resp.defect_detected is True

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            self._make_response(confidence=1.5)

        with pytest.raises(ValidationError):
            self._make_response(confidence=-0.1)

    def test_good_class_not_defect(self):
        resp = self._make_response(
            predicted_class=DefectClass.GOOD,
            defect_detected=False,
        )
        assert resp.defect_detected is False

    def test_processing_time_non_negative(self):
        with pytest.raises(ValidationError):
            self._make_response(processing_time_ms=-1.0)
