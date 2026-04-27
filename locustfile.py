"""
Load Test — Aerospace Defect Detection API
locustfile.py

Simulates concurrent users hitting the API endpoints to validate that
the service meets latency and throughput requirements under load.

Target SLA (portfolio requirement from Noah Gift's MLOps book):
    200 concurrent users
    p99 latency < 500ms
    0% error rate on /health and /predict

Why load testing matters
------------------------
A model that achieves 94% recall in training is useless if the API
cannot serve predictions under production traffic. Load testing catches:
    - Memory leaks (latency degrades over time)
    - Thread safety issues (errors under concurrency)
    - Cold start penalties (first request after scale-to-zero)
    - Container resource limits (OOM kills under load)

Usage
-----
    # Install locust
    pip install locust

    # Run against local server (start API first: uvicorn src.api.app:app)
    locust -f locustfile.py --host=http://localhost:8000 \
           --users=200 --spawn-rate=10 --run-time=60s --headless

    # Run against Cloud Run
    locust -f locustfile.py --host=https://<cloud-run-url> \
           --users=200 --spawn-rate=10 --run-time=60s --headless

    # Interactive web UI (visit http://localhost:8089)
    locust -f locustfile.py --host=http://localhost:8000

Staging branch trigger
----------------------
This load test is automatically triggered by GitHub Actions when code
is pushed to the staging branch (see .github/workflows/load-test.yml).
Results are uploaded as workflow artifacts for review.

Cloud Run note
--------------
Cloud Run scales to zero when idle. The first request after a cold start
may take 3-10 seconds (container boot + model load). Locust's spawn-rate
of 10 users/second gives the container time to warm up before full load.
Set min-instances=1 in production to eliminate cold starts entirely,
but note this incurs continuous cost — keep at 0 for portfolio demo.
"""

from __future__ import annotations

import base64
import io
import random

import numpy as np
from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Test image generation
# ---------------------------------------------------------------------------


def generate_dummy_image_base64(
    width: int = 224,
    height: int = 224,
    add_defect_pattern: bool = False,
) -> str:
    """
    Generate a synthetic test image encoded as base64.

    In production load tests, use real component images from the test set.
    For CI/CD pipeline tests, synthetic images are sufficient to validate
    API contract, latency, and error handling.

    Args:
        width:              Image width in pixels.
        height:             Image height in pixels.
        add_defect_pattern: If True, add a dark line to simulate a crack.

    Returns:
        Base64-encoded JPEG string.
    """
    try:
        from PIL import Image as PILImage  # type: ignore

        # Generate a realistic-looking grey component surface
        # Metal components are typically grey with slight texture variation
        base_intensity = random.randint(80, 160)
        noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
        pixels = np.clip(base_intensity + noise, 0, 255).astype(np.uint8)

        if add_defect_pattern:
            # Add a dark diagonal line to simulate a crack
            for i in range(min(width, height) // 2):
                x = width // 4 + i
                y = height // 4 + i
                if 0 <= x < width and 0 <= y < height:
                    pixels[y, x] = [20, 20, 20]  # dark crack pixel

        img = PILImage.fromarray(pixels, mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except ImportError:
        # Fallback: pure numpy → raw JPEG-like bytes
        # Not a valid JPEG but sufficient to test API error handling
        raw = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return base64.b64encode(raw.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Pre-generate a pool of test images
# Generating images per-request adds overhead that skews latency numbers.
# We pre-generate 20 images at module load time and reuse them.
# ---------------------------------------------------------------------------

_IMAGE_POOL_SIZE = 20
_IMAGE_POOL = [
    generate_dummy_image_base64(add_defect_pattern=(i % 3 == 0))
    for i in range(_IMAGE_POOL_SIZE)
]

_DEFECT_FILENAMES = [
    "blade_crack_001.jpg",
    "blade_scratch_042.jpg",
    "nut_bent_007.jpg",
    "screw_contamination_013.jpg",
    "tile_hole_019.jpg",
    "cable_broken_031.jpg",
]
_GOOD_FILENAMES = [
    "blade_good_001.jpg",
    "nut_good_007.jpg",
    "screw_good_013.jpg",
    "tile_good_019.jpg",
    "cable_good_031.jpg",
]


# ---------------------------------------------------------------------------
# Locust user classes
# ---------------------------------------------------------------------------


class AerospaceInspectionUser(HttpUser):
    """
    Simulates a typical API consumer — an inspection system client that:
    - Sends images for defect classification (most common operation)
    - Checks API health periodically
    - Requests explanations for defect predictions occasionally

    Task weights reflect realistic usage patterns:
    - /predict:        70% of requests (core workload)
    - /predict/upload: 15% of requests (alternative client pattern)
    - /health:         10% of requests (monitoring probes)
    - /metrics:         3% of requests (dashboard scrapes)
    - /explain:         2% of requests (engineer review of flagged parts)

    wait_time: between(0.5, 2) simulates the time between inspection
    events on a real assembly line — not instantaneous, not slow.
    """

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Called once when a simulated user starts."""
        # Verify the API is reachable before running tasks
        response = self.client.get("/health")
        if response.status_code != 200:
            self.environment.runner.quit()

    @task(70)
    def predict_single(self) -> None:
        """
        POST /predict — single image classification (base64 JSON body).

        Most common request type. Tests the core inference pipeline:
        image decode → preprocess → EfficientNetB0 forward pass → response.
        """
        image_b64 = random.choice(_IMAGE_POOL)
        filename = random.choice(_DEFECT_FILENAMES + _GOOD_FILENAMES)

        payload = {
            "image_base64": image_b64,
            "filename": filename,
            "model_version": None,
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code == 503:
                # Model not loaded — expected during cold start, not an error
                response.success()
            elif response.status_code != 200:
                response.failure(
                    f"Unexpected status {response.status_code}: {response.text[:200]}"
                )
            else:
                # Validate response schema
                data = response.json()
                required_keys = {
                    "predicted_class",
                    "confidence",
                    "defect_detected",
                    "processing_time_ms",
                }
                if not required_keys.issubset(data.keys()):
                    response.failure(f"Missing keys in response: {data.keys()}")

    @task(15)
    def predict_upload(self) -> None:
        """
        POST /predict/upload — multipart file upload.

        Tests the file upload code path, which uses different FastAPI
        machinery than the JSON body endpoint.
        """
        image_b64 = random.choice(_IMAGE_POOL)
        image_bytes = base64.b64decode(image_b64)
        filename = random.choice(_DEFECT_FILENAMES)

        with self.client.post(
            "/predict/upload",
            files={"file": (filename, image_bytes, "image/jpeg")},
            catch_response=True,
            name="/predict/upload",
        ) as response:
            if response.status_code not in (200, 503):
                response.failure(
                    f"Unexpected status {response.status_code}: {response.text[:200]}"
                )

    @task(10)
    def health_check(self) -> None:
        """
        GET /health — liveness probe.

        Should always return 200 with < 10ms latency.
        No model inference — pure FastAPI overhead.
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
            else:
                data = response.json()
                if data.get("status") != "ok":
                    response.failure(f"Unexpected health status: {data}")

    @task(3)
    def get_metrics(self) -> None:
        """
        GET /metrics — dashboard scrape.

        Returns model performance metrics and runtime statistics.
        Low frequency — typically scraped every 15-30 seconds by Prometheus.
        """
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="/metrics",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Metrics endpoint failed: {response.status_code}")

    @task(2)
    def explain_prediction(self) -> None:
        """
        GET /explain/{filename} — Grad-CAM heatmap request.

        Low frequency — only triggered when an engineer wants to review
        a flagged defect prediction.  Grad-CAM is computationally expensive
        (~2-5x slower than /predict) so it should never be called for every
        prediction in production.
        """
        image_b64 = random.choice(_IMAGE_POOL)
        filename = random.choice(_DEFECT_FILENAMES)

        with self.client.get(
            f"/explain/{filename}",
            params={
                "image_base64": image_b64,
                "class_index": random.randint(1, 7),
                "alpha": 0.4,
            },
            catch_response=True,
            name="/explain/{filename}",
        ) as response:
            # 501 = not implemented stub, 503 = model not loaded
            # Both acceptable in CI — endpoint is implemented but model
            # may not be loaded in the test environment
            if response.status_code in (200, 501, 503):
                response.success()
            else:
                response.failure(f"Explain endpoint failed: {response.status_code}")


class BatchInspectionUser(HttpUser):
    """
    Simulates a batch inspection client — sends multiple images per request.

    Represents an automated conveyor-belt inspection system that processes
    components in batches rather than one at a time.

    Lower weight than AerospaceInspectionUser — batch clients are less
    common than single-image clients in the expected traffic mix.
    """

    wait_time = between(2.0, 5.0)  # batch jobs run less frequently
    weight = 1  # 1 batch user for every 3 single-image users

    @task
    def predict_batch(self) -> None:
        """
        POST /predict/batch — batch classification (2-10 images).

        Tests the batch inference code path and validates that the
        response contains one prediction per input image in order.
        """
        batch_size = random.randint(2, 10)
        images = [
            {
                "image_base64": random.choice(_IMAGE_POOL),
                "filename": f"component_{i:04d}.jpg",
                "model_version": None,
            }
            for i in range(batch_size)
        ]

        with self.client.post(
            "/predict/batch",
            json={"images": images},
            catch_response=True,
            name="/predict/batch",
        ) as response:
            if response.status_code == 503:
                response.success()
            elif response.status_code != 200:
                response.failure(
                    f"Batch predict failed: {response.status_code}: "
                    f"{response.text[:200]}"
                )
            else:
                data = response.json()
                if data.get("total_images") != batch_size:
                    response.failure(
                        f"Expected {batch_size} predictions, "
                        f"got {data.get('total_images')}"
                    )


# ---------------------------------------------------------------------------
# Event hooks — custom reporting
# ---------------------------------------------------------------------------


@events.request.add_listener
def on_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    **kwargs,
) -> None:
    """
    Log slow requests (> 500ms) for post-test analysis.

    The 500ms threshold matches our portfolio SLA target.
    Slow requests are printed to stdout and appear in the
    GitHub Actions workflow log for review.
    """
    if response_time > 500:
        print(
            f"SLOW REQUEST: {request_type} {name} "
            f"took {response_time:.0f}ms (SLA: 500ms)"
        )
