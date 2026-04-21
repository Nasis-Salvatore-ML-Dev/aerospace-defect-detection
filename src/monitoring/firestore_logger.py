"""
Firestore Training Run Logger
src/monitoring/firestore_logger.py

Purpose: Persist every training run's metadata to Firestore so we have a
permanent, queryable audit trail of experiments — hyperparameters, metrics,
model version, dataset split sizes.  This is the MLOps "experiment registry"
pattern without the cost of a managed service.

Firestore data model
--------------------
Collection : training_runs
  Document  : <run_id>          (auto-generated UUID)
    Fields:
      run_id          str   — unique identifier for this experiment
      timestamp       str   — ISO-8601 UTC timestamp
      model_version   str   — e.g. "v1.0", "v1.1"
      status          str   — "started" | "completed" | "failed"
      dataset         dict  — split sizes and dataset path
      hyperparameters dict  — all tunable knobs used in this run
      metrics         dict  — validation metrics at end of training
      artifacts       dict  — GCS paths to saved model, ONNX, TFLite exports
      duration_seconds float — wall-clock training time

Collection : prediction_logs
  Document  : <log_id>
    Fields:
      log_id          str
      timestamp       str
      model_version   str
      image_filename  str
      predicted_class str
      confidence      float
      processing_time_ms float
      defect_detected bool

All Firestore operations are synchronous (google-cloud-firestore default client).
Cloud Run handles concurrency at the container level, so sync I/O inside a
FastAPI background task or at training time is safe and simple.

Free-tier budget: Firestore Always Free = 1 GiB storage, 50K reads/day,
20K writes/day.  A portfolio project will never come close to these limits.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

# google-cloud-firestore is installed via requirements.txt
# When running locally without GCP credentials the module will import fine
# but any network call will raise google.auth.exceptions.DefaultCredentialsError.
# We catch that at call time and log a warning so local development still works.
try:
    from google.cloud import firestore  # type: ignore

    _FIRESTORE_AVAILABLE = True
except ImportError:
    _FIRESTORE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level logger
# All log records from this module carry the name "monitoring.firestore_logger"
# which makes them easy to filter in Cloud Logging.
# ---------------------------------------------------------------------------
logger = logging.getLogger("monitoring.firestore_logger")


# ---------------------------------------------------------------------------
# Collection names — single source of truth
# ---------------------------------------------------------------------------
COLLECTION_TRAINING_RUNS = "training_runs"
COLLECTION_PREDICTION_LOGS = "prediction_logs"


# ---------------------------------------------------------------------------
# FirestoreLogger
# ---------------------------------------------------------------------------
class FirestoreLogger:
    """
    Thin wrapper around the Firestore client scoped to this project.

    Why a class rather than module-level functions?
    - The Firestore client is stateful (holds a gRPC channel).  Creating it
      once and reusing it across calls avoids per-call authentication overhead.
    - Makes unit-testing straightforward: inject a mock client via the
      constructor.

    Usage (training script):
        logger_fs = FirestoreLogger(project_id="my-gcp-project")
        run_id = logger_fs.log_run_started(
            model_version="v1.0",
            hyperparameters={"lr": 1e-4, "dropout": 0.3},
            dataset={"train": 350, "val": 75, "test": 75},
        )
        # ... train ...
        logger_fs.log_run_completed(
            run_id=run_id,
            metrics={"val_recall": 0.94, "val_accuracy": 0.91},
            artifacts={"savedmodel": "gs://bucket/models/v1.0"},
            duration_seconds=847.2,
        )

    Usage (inference / API):
        logger_fs = FirestoreLogger(project_id="my-gcp-project")
        logger_fs.log_prediction(
            model_version="v1.0",
            image_filename="blade_0042.jpg",
            predicted_class="crack",
            confidence=0.97,
            processing_time_ms=23.4,
        )
    """

    def __init__(self, project_id: str | None = None) -> None:
        """
        Initialise the Firestore client.

        Args:
            project_id: GCP project ID.  If None, the client infers it from
                        the GOOGLE_CLOUD_PROJECT env var or ADC metadata server.
        """
        self._client: Any = None  # lazy — created on first use
        self._project_id = project_id
        self._available = _FIRESTORE_AVAILABLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """
        Return (and lazily create) the Firestore client.

        Lazy initialisation means importing this module in unit tests or
        local runs without credentials does not immediately raise an error.
        """
        if not self._available:
            raise RuntimeError(
                "google-cloud-firestore is not installed. "
                "Run: pip install google-cloud-firestore"
            )
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._project_id:
                kwargs["project"] = self._project_id
            self._client = firestore.Client(**kwargs)
            logger.info(
                "Firestore client initialised (project=%s)",
                self._project_id or "inferred from environment",
            )
        return self._client

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_run_id() -> str:
        """Generate a unique run identifier."""
        return str(uuid.uuid4())

    def _write(self, collection: str, doc_id: str, data: dict[str, Any]) -> bool:
        """
        Write *data* to *collection / doc_id*.

        Returns True on success, False if Firestore is unreachable.
        Never raises — training must not crash because the metadata store
        is temporarily unavailable.
        """
        try:
            client = self._get_client()
            client.collection(collection).document(doc_id).set(data)
            logger.debug("Firestore write OK: %s/%s", collection, doc_id)
            return True
        except Exception as exc:
            # Covers: DefaultCredentialsError, GoogleAPICallError, RuntimeError
            logger.warning(
                "Firestore write failed (%s/%s): %s — continuing without persistence.",
                collection,
                doc_id,
                exc,
            )
            return False

    def _update(self, collection: str, doc_id: str, data: dict[str, Any]) -> bool:
        """
        Merge *data* into an existing document (partial update).

        Uses Firestore's update() so unchanged fields are preserved.
        """
        try:
            client = self._get_client()
            client.collection(collection).document(doc_id).update(data)
            logger.debug("Firestore update OK: %s/%s", collection, doc_id)
            return True
        except Exception as exc:
            logger.warning(
                "Firestore update failed (%s/%s): %s — continuing without persistence.",
                collection,
                doc_id,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Public API — Training runs
    # ------------------------------------------------------------------

    def log_run_started(
        self,
        model_version: str,
        hyperparameters: dict[str, Any],
        dataset: dict[str, Any],
    ) -> str:
        """
        Record that a new training run has begun.

        Call this at the very start of your training script, before model.fit().
        Returns the run_id so you can pass it to log_run_completed() later.

        Args:
            model_version:    Semantic version string, e.g. "v1.0".
            hyperparameters:  Dict of all tunable knobs for this run.
                              Example: {"learning_rate": 1e-4, "dropout": 0.3,
                                        "unfreeze_depth": 20, "batch_size": 32}
            dataset:          Dict describing the data split.
                              Example: {"train": 350, "val": 75, "test": 75,
                                        "source": "MVTec AD — aerospace subset"}

        Returns:
            run_id (str): UUID identifying this training run.
        """
        run_id = self._new_run_id()
        doc: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": self._now_iso(),
            "model_version": model_version,
            "status": "started",
            "dataset": dataset,
            "hyperparameters": hyperparameters,
            # These fields will be filled in by log_run_completed():
            "metrics": {},
            "artifacts": {},
            "duration_seconds": None,
        }
        self._write(COLLECTION_TRAINING_RUNS, run_id, doc)
        logger.info(
            "Training run started — run_id=%s  version=%s", run_id, model_version
        )
        return run_id

    def log_run_completed(
        self,
        run_id: str,
        metrics: dict[str, float],
        artifacts: dict[str, str],
        duration_seconds: float,
    ) -> None:
        """
        Update a training run document with final metrics and artifact paths.

        Call this immediately after model.fit() returns (or after export steps).

        Args:
            run_id:           The run_id returned by log_run_started().
            metrics:          Validation metrics dict.
                              Example: {"val_recall": 0.94, "val_accuracy": 0.91,
                                        "val_loss": 0.23, "val_precision": 0.89}
            artifacts:        GCS or local paths to saved artefacts.
                              Example: {"savedmodel": "gs://bucket/models/v1.0",
                                        "onnx": "gs://bucket/models/v1.0.onnx",
                                        "tflite": "gs://bucket/models/v1.0.tflite"}
            duration_seconds: Wall-clock training time in seconds.
        """
        update_data: dict[str, Any] = {
            "status": "completed",
            "metrics": metrics,
            "artifacts": artifacts,
            "duration_seconds": duration_seconds,
            "completed_at": self._now_iso(),
        }
        self._update(COLLECTION_TRAINING_RUNS, run_id, update_data)
        logger.info(
            "Training run completed — run_id=%s  val_recall=%.4f  duration=%.1fs",
            run_id,
            metrics.get("val_recall", float("nan")),
            duration_seconds,
        )

    def log_run_failed(self, run_id: str, error_message: str) -> None:
        """
        Mark a training run as failed.

        Call this inside an except block so a crash is always reflected in
        Firestore rather than leaving a "started" document with no outcome.

        Args:
            run_id:        The run_id returned by log_run_started().
            error_message: The exception message or traceback summary.
        """
        update_data: dict[str, Any] = {
            "status": "failed",
            "error_message": error_message,
            "failed_at": self._now_iso(),
        }
        self._update(COLLECTION_TRAINING_RUNS, run_id, update_data)
        logger.error("Training run FAILED — run_id=%s  error=%s", run_id, error_message)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single training run document by run_id.

        Returns None if the document does not exist or Firestore is unavailable.
        """
        try:
            client = self._get_client()
            doc = client.collection(COLLECTION_TRAINING_RUNS).document(run_id).get()
            if doc.exists:
                return doc.to_dict()
            logger.warning("Run not found in Firestore: %s", run_id)
            return None
        except Exception as exc:
            logger.warning("Firestore get failed (run_id=%s): %s", run_id, exc)
            return None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Return the *limit* most recent training runs, newest first.

        Useful for the /metrics API endpoint to surface experiment history.

        Args:
            limit: Maximum number of documents to return (default 20).

        Returns:
            List of run documents as plain dicts, sorted by timestamp descending.
        """
        try:
            client = self._get_client()
            query = (
                client.collection(COLLECTION_TRAINING_RUNS)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            return [doc.to_dict() for doc in query.stream()]
        except Exception as exc:
            logger.warning("Firestore list_runs failed: %s", exc)
            return []

    def get_best_run(self, metric: str = "val_recall") -> dict[str, Any] | None:
        """
        Return the completed run with the highest value of *metric*.

        Used to retrieve the champion model version before deployment.

        Args:
            metric: The metric key to rank by (default "val_recall").

        Returns:
            The run document with the highest metric value, or None.
        """
        runs = self.list_runs(limit=100)
        completed = [r for r in runs if r.get("status") == "completed"]
        if not completed:
            return None
        return max(
            completed,
            key=lambda r: r.get("metrics", {}).get(metric, 0.0),
        )

    # ------------------------------------------------------------------
    # Public API — Prediction logs
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        model_version: str,
        image_filename: str,
        predicted_class: str,
        confidence: float,
        processing_time_ms: float,
        defect_detected: bool | None = None,
    ) -> str:
        """
        Persist a single inference event to the prediction_logs collection.

        This feeds the drift-detection workflow: we periodically query
        prediction_logs and check whether the distribution of predicted classes
        has shifted from the training distribution.

        Args:
            model_version:      The model that served this prediction.
            image_filename:     Original filename (or UUID) of the input image.
            predicted_class:    Top-1 predicted label, e.g. "crack".
            confidence:         Softmax probability of the predicted class [0, 1].
            processing_time_ms: End-to-end inference latency in milliseconds.
            defect_detected:    Convenience boolean (True if class != "good").

        Returns:
            log_id (str): UUID of the prediction log document.
        """
        log_id = self._new_run_id()
        if defect_detected is None:
            defect_detected = predicted_class.lower() != "good"

        doc: dict[str, Any] = {
            "log_id": log_id,
            "timestamp": self._now_iso(),
            "model_version": model_version,
            "image_filename": image_filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "processing_time_ms": processing_time_ms,
            "defect_detected": defect_detected,
        }
        self._write(COLLECTION_PREDICTION_LOGS, log_id, doc)
        return log_id
