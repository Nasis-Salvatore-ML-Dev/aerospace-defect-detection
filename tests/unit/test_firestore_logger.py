"""Unit tests for src/monitoring/firestore_logger.py."""

from unittest.mock import MagicMock

from src.monitoring.firestore_logger import (
    COLLECTION_PREDICTION_LOGS,
    COLLECTION_TRAINING_RUNS,
    FirestoreLogger,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_logger_with_mock_client() -> tuple[FirestoreLogger, MagicMock]:
    """
    Return a FirestoreLogger with a fully mocked Firestore client.

    We never hit real Firestore in unit tests — that would require
    GCP credentials and network access. The mock intercepts all
    client.collection().document().set() / .update() calls.
    """
    logger = FirestoreLogger(project_id="test-project")

    mock_client = MagicMock()
    logger._client = mock_client
    logger._available = True

    return logger, mock_client


# ---------------------------------------------------------------------------
# FirestoreLogger._write
# ---------------------------------------------------------------------------


class TestFirestoreLoggerWrite:
    def test_write_calls_firestore_set(self):
        logger, mock_client = make_logger_with_mock_client()
        result = logger._write("test_collection", "doc_id_123", {"key": "value"})

        assert result is True
        mock_client.collection.assert_called_once_with("test_collection")
        mock_client.collection().document.assert_called_once_with("doc_id_123")
        mock_client.collection().document().set.assert_called_once_with(
            {"key": "value"}
        )

    def test_write_returns_false_on_exception(self):
        logger, mock_client = make_logger_with_mock_client()
        mock_client.collection.side_effect = Exception("Firestore unavailable")

        result = logger._write("collection", "doc_id", {"key": "value"})
        assert result is False

    def test_write_does_not_raise_on_exception(self):
        """Training must not crash if Firestore is unavailable."""
        logger, mock_client = make_logger_with_mock_client()
        mock_client.collection.side_effect = RuntimeError("network error")

        # Should not raise
        logger._write("collection", "doc_id", {})


# ---------------------------------------------------------------------------
# FirestoreLogger.log_run_started
# ---------------------------------------------------------------------------


class TestLogRunStarted:
    def test_returns_run_id_string(self):
        logger, _ = make_logger_with_mock_client()
        run_id = logger.log_run_started(
            model_version="v1.0",
            hyperparameters={"lr": 1e-3},
            dataset={"train": 350, "val": 75},
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID4 format

    def test_run_id_is_unique(self):
        logger, _ = make_logger_with_mock_client()
        ids = {
            logger.log_run_started(
                model_version="v1.0",
                hyperparameters={},
                dataset={},
            )
            for _ in range(10)
        }
        assert len(ids) == 10  # all unique

    def test_document_written_to_correct_collection(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_run_started(
            model_version="v1.0",
            hyperparameters={"dropout": 0.3},
            dataset={"train": 350},
        )
        mock_client.collection.assert_called_with(COLLECTION_TRAINING_RUNS)

    def test_document_contains_required_fields(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_run_started(
            model_version="v1.0",
            hyperparameters={"lr": 1e-3, "dropout": 0.3},
            dataset={"train": 350, "val": 75},
        )

        call_args = mock_client.collection().document().set.call_args[0][0]
        required_fields = {
            "run_id",
            "timestamp",
            "model_version",
            "status",
            "dataset",
            "hyperparameters",
            "metrics",
            "artifacts",
        }
        assert required_fields.issubset(call_args.keys())

    def test_initial_status_is_started(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_run_started(
            model_version="v1.0",
            hyperparameters={},
            dataset={},
        )
        call_args = mock_client.collection().document().set.call_args[0][0]
        assert call_args["status"] == "started"


# ---------------------------------------------------------------------------
# FirestoreLogger.log_run_completed
# ---------------------------------------------------------------------------


class TestLogRunCompleted:
    def test_updates_correct_document(self):
        logger, mock_client = make_logger_with_mock_client()
        run_id = "test-run-id-123"
        logger.log_run_completed(
            run_id=run_id,
            metrics={"val_recall": 0.94},
            artifacts={"savedmodel": "gs://bucket/v1.0"},
            duration_seconds=847.2,
        )
        mock_client.collection().document.assert_called_with(run_id)

    def test_status_set_to_completed(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_run_completed(
            run_id="run_id",
            metrics={"val_recall": 0.94},
            artifacts={},
            duration_seconds=100.0,
        )
        call_args = mock_client.collection().document().update.call_args[0][0]
        assert call_args["status"] == "completed"

    def test_metrics_and_artifacts_stored(self):
        logger, mock_client = make_logger_with_mock_client()
        metrics = {"val_recall": 0.94, "val_accuracy": 0.91}
        artifacts = {"savedmodel": "gs://bucket/v1.0", "onnx": "gs://bucket/v1.0.onnx"}

        logger.log_run_completed(
            run_id="run_id",
            metrics=metrics,
            artifacts=artifacts,
            duration_seconds=500.0,
        )
        call_args = mock_client.collection().document().update.call_args[0][0]
        assert call_args["metrics"] == metrics
        assert call_args["artifacts"] == artifacts


# ---------------------------------------------------------------------------
# FirestoreLogger.log_run_failed
# ---------------------------------------------------------------------------


class TestLogRunFailed:
    def test_status_set_to_failed(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_run_failed(run_id="run_id", error_message="OOM error")
        call_args = mock_client.collection().document().update.call_args[0][0]
        assert call_args["status"] == "failed"
        assert "OOM error" in call_args["error_message"]


# ---------------------------------------------------------------------------
# FirestoreLogger.log_prediction
# ---------------------------------------------------------------------------


class TestLogPrediction:
    def test_returns_log_id(self):
        logger, _ = make_logger_with_mock_client()
        log_id = logger.log_prediction(
            model_version="v1.0",
            image_filename="blade_042.jpg",
            predicted_class="crack",
            confidence=0.94,
            processing_time_ms=28.4,
        )
        assert isinstance(log_id, str)
        assert len(log_id) == 36

    def test_written_to_prediction_logs_collection(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_prediction(
            model_version="v1.0",
            image_filename="blade_042.jpg",
            predicted_class="crack",
            confidence=0.94,
            processing_time_ms=28.4,
        )
        mock_client.collection.assert_called_with(COLLECTION_PREDICTION_LOGS)

    def test_defect_detected_inferred_from_class(self):
        logger, mock_client = make_logger_with_mock_client()

        # Crack → defect_detected should be True
        logger.log_prediction(
            model_version="v1.0",
            image_filename="blade.jpg",
            predicted_class="crack",
            confidence=0.94,
            processing_time_ms=28.4,
        )
        call_args = mock_client.collection().document().set.call_args[0][0]
        assert call_args["defect_detected"] is True

    def test_good_class_not_defect(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_prediction(
            model_version="v1.0",
            image_filename="blade.jpg",
            predicted_class="good",
            confidence=0.97,
            processing_time_ms=25.1,
        )
        call_args = mock_client.collection().document().set.call_args[0][0]
        assert call_args["defect_detected"] is False

    def test_explicit_defect_detected_override(self):
        logger, mock_client = make_logger_with_mock_client()
        logger.log_prediction(
            model_version="v1.0",
            image_filename="blade.jpg",
            predicted_class="good",
            confidence=0.97,
            processing_time_ms=25.1,
            defect_detected=True,  # explicit override
        )
        call_args = mock_client.collection().document().set.call_args[0][0]
        assert call_args["defect_detected"] is True


# ---------------------------------------------------------------------------
# FirestoreLogger.get_best_run
# ---------------------------------------------------------------------------


class TestGetBestRun:
    def test_returns_run_with_highest_metric(self):
        logger, mock_client = make_logger_with_mock_client()

        runs = [
            {"run_id": "a", "status": "completed", "metrics": {"val_recall": 0.85}},
            {"run_id": "b", "status": "completed", "metrics": {"val_recall": 0.94}},
            {"run_id": "c", "status": "completed", "metrics": {"val_recall": 0.91}},
        ]

        mock_docs = [MagicMock() for _ in runs]
        for doc, run in zip(mock_docs, runs):
            doc.to_dict.return_value = run

        mock_client.collection().order_by().limit().stream.return_value = mock_docs

        best = logger.get_best_run(metric="val_recall")
        assert best["run_id"] == "b"

    def test_returns_none_when_no_completed_runs(self):
        logger, mock_client = make_logger_with_mock_client()

        runs = [
            {"run_id": "a", "status": "started", "metrics": {}},
        ]
        mock_docs = [MagicMock()]
        mock_docs[0].to_dict.return_value = runs[0]
        mock_client.collection().order_by().limit().stream.return_value = mock_docs

        best = logger.get_best_run()
        assert best is None
