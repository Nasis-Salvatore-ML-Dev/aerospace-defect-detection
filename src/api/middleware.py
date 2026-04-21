"""
API Request Logging Middleware
src/api/middleware.py

Purpose: Intercept every HTTP request/response that passes through the FastAPI
application and produce structured log records at three destinations:

    STDERR  — ERROR level and above  (container runtime captures this)
    STDOUT  — INFO level and above   (Cloud Logging ingests this)
    File    — ALL levels             (local debug, mounted volume or GCS sync)

This satisfies two portfolio requirements from Noah Gift's MLOps book:
  1. "Add Python logging to a script that will log errors to STDERR,
     info statements to STDOUT, and all levels to a file."
  2. Production-grade observability: every request is timed and logged so
     latency regressions are immediately visible in Cloud Logging.

Architecture note
-----------------
FastAPI supports two middleware styles:
  a) Starlette BaseHTTPMiddleware   — simple, class-based, slight overhead
  b) @app.middleware("http")        — decorator, slightly lower overhead

We use BaseHTTPMiddleware because it is easier to unit-test (inject a mock
app) and easier for a reader to understand the request/response lifecycle.

The middleware does NOT touch business logic — it is purely cross-cutting
infrastructure.  Every route function stays clean.

Logging destinations
--------------------
Handler          Level     Destination   Format
---------------  --------  ------------  -------
StreamHandler    INFO+     STDOUT        JSON
StreamHandler    ERROR+    STDERR        JSON
RotatingFile     DEBUG+    logs/api.log  JSON   (rotates at 10 MB, keeps 5)

JSON format makes records machine-parseable by Cloud Logging without any
additional agent configuration — Cloud Run streams stdout/stderr to Cloud
Logging automatically.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def configure_logging(
    log_dir: str | Path = "logs",
    log_filename: str = "api.log",
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.ERROR,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure the root logger with three handlers:
      - STDOUT  : INFO and above
      - STDERR  : ERROR and above
      - File    : DEBUG and above (rotating, 10 MB max, 5 backups)

    Call this ONCE at application startup (inside the FastAPI lifespan handler).
    All subsequent getLogger() calls in any module will inherit this config.

    Args:
        log_dir:       Directory for the log file (created if it does not exist).
        log_filename:  Name of the rotating log file.
        stdout_level:  Minimum level sent to STDOUT (default INFO).
        stderr_level:  Minimum level sent to STDERR (default ERROR).
        file_level:    Minimum level written to file  (default DEBUG).

    Returns:
        The configured root logger.
    """
    log_path = Path(log_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # let handlers decide what they accept

    # Remove any handlers added by previous calls (e.g. during testing)
    root_logger.handlers.clear()

    formatter = _JsonFormatter()

    # ------------------------------------------------------------------
    # Handler 1: STDOUT — INFO and above
    # Cloud Run streams this to Cloud Logging automatically.
    # ------------------------------------------------------------------
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)
    stdout_handler.addFilter(_MaxLevelFilter(stderr_level - 1))  # exclude ERROR+
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # ------------------------------------------------------------------
    # Handler 2: STDERR — ERROR and above
    # Cloud Run treats stderr as the error stream; Cloud Logging marks
    # these records with severity=ERROR automatically.
    # ------------------------------------------------------------------
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    # ------------------------------------------------------------------
    # Handler 3: Rotating file — all levels
    # Useful during local development and CI runs.  In production on Cloud
    # Run you would mount a GCS FUSE volume or simply rely on Cloud Logging.
    # ------------------------------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "stdout_level": logging.getLevelName(stdout_level),
            "stderr_level": logging.getLevelName(stderr_level),
            "log_file": str(log_path),
        },
    )
    return root_logger


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object.

    Cloud Logging parses these records automatically when the container
    writes to stdout/stderr without any additional agent configuration.

    Fields emitted:
      timestamp  — ISO-8601 UTC
      severity   — Cloud Logging severity name (INFO, WARNING, ERROR …)
      logger     — logger name (module path)
      message    — the formatted log message
      **extra    — any extra fields passed via logging.info(..., extra={})
    """

    # Map Python level names to Cloud Logging severity strings
    _SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Base payload
        payload: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f"),
            "severity": self._SEVERITY_MAP.get(record.levelname, record.levelname),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Attach any extra fields injected via extra={} in the logging call
        _standard_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _standard_keys:
                payload[key] = value

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Helper: filter records BELOW a maximum level
# ---------------------------------------------------------------------------


class _MaxLevelFilter(logging.Filter):
    """
    Accept only records whose level is <= max_level.

    Used to prevent INFO/DEBUG records from appearing on STDERR
    (which is reserved for ERROR and above).
    """

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


# ---------------------------------------------------------------------------
# Request / Response logging middleware
# ---------------------------------------------------------------------------

logger = logging.getLogger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that logs every HTTP request and response.

    What it records:
      - On request  arrival : method, path, client IP, a unique request_id
      - On response dispatch: status code, latency in milliseconds

    The request_id (UUID4) is injected into the response headers as
    X-Request-ID so clients and support staff can correlate a specific
    request in Cloud Logging.

    Why latency matters for this project
    -------------------------------------
    The portfolio brief specifies deployment constraints: latency benchmarking
    for ONNX / TFLite formats.  Logging response time on every request means
    we have a continuous latency distribution in Cloud Logging — not just
    a one-shot benchmark figure.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Assign a unique ID to this request for end-to-end tracing
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Log the incoming request at INFO level → goes to STDOUT + file
        logger.info(
            "Request received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": _get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
            },
        )

        # Call the actual route handler (or the next middleware in the chain)
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            # Unhandled exception — log at ERROR level → goes to STDERR + file
            logger.error(
                "Unhandled exception during request",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise exc  # re-raise so FastAPI returns a 500

        elapsed_ms = (time.perf_counter() - start_time) * 1_000

        # Log the outgoing response
        _log_response(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            elapsed_ms=elapsed_ms,
        )

        # Inject the request ID into the response so clients can use it
        response.headers["X-Request-ID"] = request_id
        return response


def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP.

    Cloud Run sits behind Google's load balancer which sets the
    X-Forwarded-For header.  We prefer that over request.client.host
    which would be the load balancer's internal IP.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list; the first entry is the client
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _log_response(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    elapsed_ms: float,
) -> None:
    """
    Emit the response log record at the appropriate level.

    5xx responses → ERROR  (appears on STDERR and triggers Cloud Monitoring alerts)
    4xx responses → WARNING
    2xx/3xx       → INFO
    """
    extra = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "elapsed_ms": round(elapsed_ms, 2),
    }

    if status_code >= 500:
        logger.error("Response dispatched — server error", extra=extra)
    elif status_code >= 400:
        logger.warning("Response dispatched — client error", extra=extra)
    else:
        logger.info("Response dispatched", extra=extra)
