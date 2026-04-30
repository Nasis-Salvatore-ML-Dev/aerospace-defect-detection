"""
Load test for the Aerospace Defect Detection API.

Tests the /predict endpoint with synthetic JPEG image payloads,
measuring p50/p95/p99 latency and throughput under concurrent load.

Usage (local):
    locust -f tests/load/locustfile.py --headless \
        -u 10 -r 2 -t 60s \
        --host https://defect-api-staging-xyz.run.app \
        --html reports/load_test_report.html

Usage (CI — triggered automatically on push to staging branch):
    See .github/workflows/load-test.yml
"""

import io
import random
import struct
import zlib

from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Synthetic image factory
# ---------------------------------------------------------------------------


def _make_minimal_png(
    width: int = 8,
    height: int = 8,
    r: int = 128,
    g: int = 128,
    b: int = 128,
) -> bytes:
    """
    Build a valid minimal PNG image in pure Python — no external dependencies.

    The preprocessing pipeline resizes to (224, 224), so the input dimensions
    do not matter for correctness; we keep it small (8x8) to minimize network
    overhead during the load test.
    """

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR: width, height, bit depth=8, color type=2 (RGB)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _png_chunk(b"IHDR", ihdr_data)

    # IDAT: raw scanlines, each prefixed with filter byte 0x00 (None)
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + bytes([r, g, b] * width)
    compressed = zlib.compress(raw_rows)
    idat = _png_chunk(b"IDAT", compressed)

    # IEND
    iend = _png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Image pool — generated once at startup, reused across requests
# ---------------------------------------------------------------------------

_IMAGE_POOL_SIZE = 20
_IMAGE_POOL: list[bytes] = []


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Pre-generate synthetic images before the test starts."""
    global _IMAGE_POOL
    _IMAGE_POOL = [_make_minimal_png() for _ in range(_IMAGE_POOL_SIZE)]
    print(f"[locust] Image pool ready: {_IMAGE_POOL_SIZE} synthetic PNGs")


# ---------------------------------------------------------------------------
# Load test user
# ---------------------------------------------------------------------------


class DefectDetectionUser(HttpUser):
    """
    Simulates a client submitting component images for defect inspection.

    Wait time of 1-3 seconds models a realistic inspection cadence —
    not a flood, but sustained concurrent load.
    """

    wait_time = between(1, 3)

    # -------------------------------------------------------------------
    # Tasks — weighted by realistic usage patterns
    # -------------------------------------------------------------------

    @task(10)
    def predict_single(self) -> None:
        """
        POST /predict with a synthetic PNG.

        Weight 10 = 83% of traffic. This is the hot path — the endpoint
        that must satisfy p95 < 500ms under load.
        """
        image_bytes = random.choice(_IMAGE_POOL)
        image_file = io.BytesIO(image_bytes)

        with self.client.post(
            "/predict",
            files={"file": ("component.png", image_file, "image/png")},
            catch_response=True,
            name="/predict",
        ) as response:
            self._validate_predict_response(response)

    @task(2)
    def health_check(self) -> None:
        """
        GET /health.

        Weight 2 = 17% of traffic. Load balancers and orchestrators
        poll this continuously; it must return 200 under full load.
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"/health returned {response.status_code}: {response.text[:200]}"
                )
            else:
                response.success()

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _validate_predict_response(self, response) -> None:
        """
        Mark the response as success or failure based on:
        1. HTTP status code must be 200.
        2. Response body must contain required fields.
        3. Confidence must be a float in [0, 1].
        4. Label must be one of the expected classes.
        """
        if response.status_code != 200:
            response.failure(
                f"Expected 200, got {response.status_code}: {response.text[:200]}"
            )
            return

        try:
            body = response.json()
        except Exception as exc:
            response.failure(f"Response is not valid JSON: {exc}")
            return

        required_fields = {"label", "confidence", "model_version", "latency_ms"}
        missing = required_fields - body.keys()
        if missing:
            response.failure(f"Missing fields in response: {missing}")
            return

        confidence = body.get("confidence", -1)
        if not (0.0 <= confidence <= 1.0):
            response.failure(f"Confidence out of range: {confidence}")
            return

        valid_labels = {"normal", "defective"}
        label = body.get("label", "")
        if label not in valid_labels:
            response.failure(f"Unexpected label: {label!r}")
            return

        response.success()
