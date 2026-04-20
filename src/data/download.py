"""MVTec Anomaly Detection dataset download and verification script."""

import logging
import sys
import tarfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MVTEC_URL = (
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282"
    "/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
)

# Aerospace-relevant categories only (keeps training time manageable on Colab)
AEROSPACE_CATEGORIES = [
    "metal_nut",  # → fastener
    "screw",  # → structural bolt
    "tile",  # → thermal shield panel
    "cable",  # → wiring harness
    "capsule",  # → pressure vessel component
]

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_dataset(output_dir: Path = DATA_DIR) -> Path:
    """Download MVTec AD dataset.

    Args:
        output_dir: Directory to save the downloaded archive.

    Returns:
        Path to the downloaded archive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "mvtec_anomaly_detection.tar.xz"

    if archive_path.exists():
        logger.info("Archive already exists at %s — skipping download.", archive_path)
        return archive_path

    logger.info("Downloading MVTec AD dataset...")
    logger.info("URL: %s", MVTEC_URL)

    response = requests.get(MVTEC_URL, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                print(f"\rProgress: {pct:.1f}%", end="", flush=True)

    print()
    logger.info("Download complete: %s", archive_path)
    return archive_path


def extract_categories(
    archive_path: Path,
    categories: list[str] = AEROSPACE_CATEGORIES,
    output_dir: Path = DATA_DIR,
) -> None:
    """Extract only aerospace-relevant categories from the archive.

    Args:
        archive_path: Path to the downloaded .tar.xz archive.
        categories: List of MVTec categories to extract.
        output_dir: Directory to extract into.
    """
    logger.info("Extracting categories: %s", categories)

    with tarfile.open(archive_path, "r:xz") as tar:
        members = [
            m
            for m in tar.getmembers()
            if any(m.name.startswith(cat) for cat in categories)
        ]
        logger.info("Extracting %d files...", len(members))
        tar.extractall(path=output_dir, members=members)

    logger.info("Extraction complete → %s", output_dir)


def verify_structure(data_dir: Path = DATA_DIR) -> bool:
    """Verify that expected category directories exist.

    Args:
        data_dir: Root data directory to verify.

    Returns:
        True if all expected categories are present.
    """
    missing = []
    for category in AEROSPACE_CATEGORIES:
        cat_path = data_dir / category
        if not cat_path.exists():
            missing.append(category)

    if missing:
        logger.error("Missing categories: %s", missing)
        return False

    logger.info("Dataset structure verified. All categories present.")
    return True


if __name__ == "__main__":
    archive = download_dataset()
    extract_categories(archive)
    verify_structure()
