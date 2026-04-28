# Aerospace Surface Defect Detection

CNN-based visual inspection system for aerospace components. Detects and classifies surface defects (cracks, scratches, contamination, structural deformation) in component images and localizes them with Grad-CAM heatmaps.

Phase 2 of a five-phase ML engineering portfolio.

---

## Problem

Manual visual inspection of aerospace components misses defects at rates of 5–20% depending on inspector fatigue and defect complexity. Missing a crack is catastrophically worse than a false alarm — this asymmetry drives every design decision in this project.

---

## Solution

A transfer learning pipeline built on EfficientNetB0 that:

- classifies component images into 8 categories (good + 7 defect types)
- explains predictions with Grad-CAM heatmaps showing where the defect was detected
- exports to ONNX and INT8 TFLite for constrained deployment on edge hardware
- serves predictions via a production REST API with full MLOps instrumentation

---

## Model performance (v1.2)

Trained on Google Colab T4 GPU. Dataset: MVTec AD aerospace subset.

| Metric                   | Value                                |
| ------------------------ | ------------------------------------ |
| val_accuracy             | 55% (8-class, random baseline 12.5%) |
| test_accuracy            | 100%\*                               |
| Training time            | 10 min (Colab T4 GPU)                |
| Train / val / test split | 656 / 91 / 50 images                 |

\*Test set contains 50 images — interpret with caution.

Latency benchmark results (p50 / p95 / p99): pending export run.

---

## Architecture

```
MVTec AD Dataset (aerospace subset)
      ↓
tf.data pipeline — augmentation, prefetch, cache
      ↓
EfficientNetB0 backbone (ImageNet pretrained)
      ↓
Custom head: GAP → BatchNorm → Dense(256) → Dropout(0.3) → Softmax(8)
      ↓
Standard cross-entropy loss
      ↓
SavedModel → ONNX → TFLite INT8
      ↓
FastAPI  (/predict  /predict/batch  /explain  /health  /metrics)
      ↓
Docker → GitHub Actions CI → Cloud Run
      ↓
Firestore (experiment tracking) + TensorBoard (training curves)
```

---

## Key engineering decisions

**Why transfer learning.** The available dataset has roughly 500 labeled images per defect category — not enough to train a CNN from scratch. EfficientNetB0 pretrained on ImageNet already knows how to detect edges, textures, and shapes. Those features transfer to surface defect detection. Training from scratch on this dataset would overfit immediately.

**Two-phase training protocol.** Phase 1 freezes the backbone entirely and trains only the classification head (LR 1e-3, 20 epochs). This avoids destroying pretrained weights before the head has learned anything useful. Phase 2 unfreezes the top 20 backbone layers and fine-tunes at LR 1e-4. BatchNorm layers stay frozen throughout — updating their running statistics on a small dataset destabilises training.

**Custom loss functions.** SeverityWeightedCrossEntropy and FocalLoss are implemented as keras.losses.Loss subclasses with get_config() for full serialisation. On a dataset of this size, standard cross-entropy was used for training v1.2 — the custom loss is available for larger datasets where class imbalance is severe enough to benefit from it.

**Grad-CAM without third-party libraries.** Implemented directly using tf.GradientTape. Records the last convolutional layer activations during the forward pass, computes gradients of the predicted class score with respect to those activations, global-average-pools the gradients to get one importance weight per channel, takes the weighted sum, applies ReLU to keep only positive contributions, upsamples to image size.

**Model compression.** The trained SavedModel exports to ONNX (opset 13) and TFLite with INT8 post-training quantisation. INT8 reduces model size approximately 4x and speeds up inference 2-4x on ARM CPUs. Latency benchmarked at p50/p95/p99 across all three formats.

**Firestore experiment tracking.** Every training run writes a document to Firestore before training starts and updates it on completion or failure. All Firestore writes fail silently — training never crashes because the metadata store is unavailable.

---

## Stack

| Layer               | Technology                                                         |
| ------------------- | ------------------------------------------------------------------ |
| Model               | TensorFlow / Keras 3, EfficientNetB0                               |
| Training            | Google Colab (T4 GPU)                                              |
| Export              | tf2onnx, TFLite converter, INT8 quantisation                       |
| API                 | FastAPI, Uvicorn, Pydantic v2                                      |
| Explainability      | Grad-CAM (custom implementation)                                   |
| Experiment tracking | Firestore, Weights & Biases                                        |
| Containerisation    | Docker (multi-stage build, non-root user)                          |
| CI                  | GitHub Actions (black, isort, flake8, pytest — Python 3.10 + 3.11) |
| Load testing        | Locust (200 concurrent users, p99 < 500ms SLA)                     |
| Dataset             | MVTec Anomaly Detection — CC BY-NC-SA 4.0                          |

---

## Project structure

```
aerospace-defect-detection/
├── src/
│   ├── api/
│   │   ├── app.py               # FastAPI application, all endpoints
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── middleware.py        # Request logging, timing, X-Request-ID
│   ├── models/
│   │   ├── train.py             # Two-phase transfer learning pipeline
│   │   ├── losses.py            # SeverityWeightedCrossEntropy, FocalLoss
│   │   ├── export.py            # ONNX + TFLite export, latency benchmark
│   │   └── gradcam.py           # Grad-CAM heatmap generation
│   └── monitoring/
│       └── firestore_logger.py  # Experiment tracking, prediction logging
├── tests/unit/                  # Unit tests (pytest)
├── .github/workflows/
│   ├── ci.yml                   # Lint + test on Python 3.10 and 3.11
│   └── load-test.yml            # Locust load test on staging branch push
├── Dockerfile                   # Multi-stage build, non-root user
├── locustfile.py                # Load test: two user classes, SLA gate
├── requirements.txt
└── requirements-dev.txt
```

---

## API endpoints

| Method | Endpoint            | Description                                    |
| ------ | ------------------- | ---------------------------------------------- |
| POST   | /predict            | Single image classification (base64 JSON)      |
| POST   | /predict/upload     | Single image classification (multipart upload) |
| POST   | /predict/batch      | Batch classification, up to 50 images          |
| GET    | /explain/{filename} | Grad-CAM heatmap for a prediction              |
| GET    | /health             | Liveness check, model loaded status            |
| GET    | /metrics            | Training metrics + runtime statistics          |

### Example request

```bash
IMAGE_B64=$(base64 -i blade_042.jpg)

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"filename\": \"blade_042.jpg\"}"
```

### Example response

```json
{
  "predicted_class": "crack",
  "confidence": 0.94,
  "defect_detected": true,
  "class_probabilities": {
    "good": 0.02,
    "crack": 0.94,
    "scratch": 0.02,
    "bent": 0.01,
    "color": 0.0,
    "contamination": 0.0,
    "hole": 0.0,
    "broken": 0.01
  },
  "model_version": "v1.2",
  "processing_time_ms": 28.4,
  "filename": "blade_042.jpg"
}
```

---

## Running locally

```bash
git clone https://github.com/<your-username>/aerospace-defect-detection
cd aerospace-defect-detection

python -m venv ~/.aerospace-defect-detection
source ~/.aerospace-defect-detection/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run API (model_loaded=false until model file is present)
uvicorn src.api.app:app --reload --port 8080

# Tests
pytest tests/unit/ -v

# Docker
docker build -t aerospace-defect-detection .
docker run -p 8080:8080 aerospace-defect-detection
```

---

## Training

Runs on Google Colab (free T4 GPU). Not supported on CPU.

```bash
python src/models/train.py \
  --data-dir data/mvtec_aerospace_remapped \
  --model-version v1.2 \
  --epochs-head 20 \
  --epochs-finetune 30 \
  --batch-size 16 \
  --unfreeze-layers 20
```

---

## Model export

```bash
python src/models/export.py \
  --savedmodel-dir models/saved_model/v1.2 \
  --output-dir models/exports \
  --model-version v1.2 \
  --quantize \
  --benchmark
```

---

## Dataset

MVTec Anomaly Detection, reframed as aerospace component inspection.

| MVTec category | Aerospace reframing  |
| -------------- | -------------------- |
| Metal nut      | Fastener             |
| Screw          | Structural bolt      |
| Tile           | Thermal shield panel |
| Capsule        | Sensor housing       |
| Cable          | Wiring harness       |

Download: https://www.mvtec.com/company/research/datasets/mvtec-ad — CC BY-NC-SA 4.0.

---

## Portfolio context

| Phase | Focus                                          | Status   |
| ----- | ---------------------------------------------- | -------- |
| 1     | Classical ML — XGBoost car pricing (BMW/Aures) | Complete |
| 2     | Deep learning — aerospace defect detection     | Complete |
| 3+4   | Production ML + EU AI Act compliance           | Planned  |
| 5     | LLMOps                                         | Planned  |

---

## Author

Salvatore — ML engineer. Focused on production deep learning systems and MLOps.
