"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Aerospace Defect Detection API",
    description="CNN-based defect detection for aerospace components.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
