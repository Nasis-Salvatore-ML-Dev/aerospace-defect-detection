### Dataset web address

https://ww

# Enhanced End-to-End ML Workflow: BMW Pricing System

## Project Overview

### Purpose of the Project

This project's purpose is to build a machine learning system that predicts used BMW vehicle prices to replace manual pricing by dealership specialists.
The system delivers predictions via REST API with sub-50ms latency, achieving MAE < €2,500 across all BMW segments.

### The current situation

BMW dealerships face a critical business problem: pricing used vehicles accurately in a volatile market.
This problem is determined by the fact that dealerships rely on manual pricing based on book values, competitor analysis, and gut instinct.

### Impact so far

This problem causes a 25% of inventory to be **underpriced** or **overpriced** by more than 5%.
Because the market consists of 2400 cars/year, this means 600 cars are mispriced during a year - for a total
cost of €804K/year - and only 1800 cars are safely priced.

## Objective

Therefore, the **business objective** is to decrease €804K/year by reducing the error with which 25% of cars are mispriced.

### M1. Business Metrics

**Financial Targets:**

| Metric                      | Current | Target | Annual Savings |
| --------------------------- | ------- | ------ | -------------- |
| Lost margin (underpricing)  | €480K   | €288K  | €192K          |
| Holding costs (overpricing) | €324K   | €194K  | €130K          |
| **Total Annual Savings**    | €804K   | €482K  | **€322K**      |

**Monthly Monitoring Targets:**

### M2. Model Performance

- MAE (Mean Absolute Error): €1500 - €2500
  _Meaning_: Average absolute error across all cars
- MAPE (Mean Absolute Percentage Error): 3.5% - 4.5%
  _Meaning_: Average percent error across all cars
- R^2: >=0.85, which measures how well the model's features capture most of the factors that drive the price

- RMSE: €2,200 - €3,000. RMSE penalizes large prediction errors

**Technical Performance Targets:**

| Metric                             | Target (Good) | Max Acceptable | Business Meaning                                     |
| ---------------------------------- | ------------- | -------------- | ---------------------------------------------------- |
| **MAE** (Mean Absolute Error)      | < €1500       | < €2500        | Average price prediction error                       |
| **RMSE** (Root Mean Squared Error) | < 2200        | < €3000        | Penalizes large errors more heavily                  |
| **R²** (R-squared)                 | > 0.88        | > 0.85         | % of price variance the model explains               |
| **MAPE** (Mean Absolute % Error)   | < 3.5%        | < 4.5%         | Scale-independent error (works for all price ranges) |

---

### M3. API Performance Metrics (System Speed & Reliability)

| Metric          | Target (Good) | Max Acceptable | What It Measures                                                |
| --------------- | ------------- | -------------- | --------------------------------------------------------------- |
| **p50 Latency** | < 20ms        | < 30ms         | Typical response time (50% of requests)                         |
| **p95 Latency** | < 50ms        | < 100ms        | 95% of requests meet this SLA                                   |
| **p99 Latency** | < 100ms       | < 200ms        | Worst-case response time (1 in 100 requests feels slow)         |
| **API Uptime**  | > 99.9%       | > 99.5%        | Service availability (downtime < 8 hours/year)                  |
| **Error Rate**  | < 0.1%        | < 0.5%         | Failed predictions (validation or server errors) 1 in 1000 fail |
| **Throughput**  | > 100 req/sec | > 50 req/sec   | How many concurrent requests we can handle                      |

**Why These Targets?**

---

### M4. Segment Performance Metrics (Quality Assurance by Car Type)

| Car Price Segment       | MAE Target | Rationale                                                                  |
| ----------------------- | ---------- | -------------------------------------------------------------------------- |
| **Economy** (< €20K)    | < €1,500   | ~10% of a €15K car. Lower absolute errors acceptable for cheaper cars      |
| **Mid-Range** (€20-35K) | < €2,500   | ~9% of a €27.5K car. Core BMW segment - standard target                    |
| **Premium** (€35-50K)   | < €3,000   | ~7% of a €42.5K car. Higher value justifies slightly higher absolute error |
| **Luxury** (> €50K)     | < €4,000   | <8% of a €50K+ car. Rare, high-variance models - relaxed threshold         |

---

### M5. Data Quality & Drift Monitoring to protect the business metrics (ensure that €322K are indeed saved)

| Metric                                   | Baseline Value | Alert Threshold                         | What To Do                                             |
| ---------------------------------------- | -------------- | --------------------------------------- | ------------------------------------------------------ |
| **Avg Input Mileage**                    | 45,000 km      | ±30% shift (< 31,500 km or > 58,500 km) | Data distribution changed - investigate source         |
| **Avg Input Price** (actual)             | €35,000        | ±20% shift (< €28,000 or > €42,000)     | Market shift or data quality issue                     |
| **Avg Model Prediction**                 | €35,000        | ±20% shift (< €28,000 or > €42,000)     | Model drift detected - consider retraining             |
| **Feature Distribution** (KL Divergence) | < 0.2          | > 0.3                                   | Significant drift - retraining needed                  |
| **% Unknown Model Keys**                 | < 5%           | > 10%                                   | New BMW models appearing that weren't in training data |

**What Is KL Divergence?**

**Example Drift Scenario:**

```
Week 1:  Avg mileage = 45,000 km ✅
Week 2:  Avg mileage = 48,000 km ✅ (+6.7%, within ±30%)
Week 3:  Avg mileage = 52,000 km ⚠️  (+15.6%, monitor closely)
Week 4:  Avg mileage = 62,000 km ❌ (+37.8%, ALERT!)

Action: Investigate why high-mileage cars are suddenly appearing
Possible causes:
- Data source changed (now including fleet vehicles)
- Market shift (more used cars entering market)
- Data quality issue (mileage field corrupted)
```

---

### M6. Retraining Triggers (When to update the model and to prevent unnecessary retraining from a single noisy signal)

**Retrain if 2 or more of these conditions are met:**

## |

## Summary: Critical Metrics at a Glance

### ✅ Model is HEALTHY when:

### ⚠️ Model needs ATTENTION when:

### ❌ Model needs IMMEDIATE ACTION when:

---

# Create the project scaffolding and create a new GitHub repository

### Step 1 — Create the GitHub repository

Name: `aerospace-defect-detection`

- Public repo
- No README, no .gitignore yet — you will add these manually
- Clone it locally

---

### Step 2 — Build this directory structure

```
aerospace-defect-detection/
├── .github/
│   └── workflows/          # empty for now
├── src/
│   ├── api/
│   ├── data/
│   ├── models/
│   ├── monitoring/
│   └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
├── config/
├── data/
│   └── golden/
├── docs/
├── scripts/
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .flake8
└── .pre-commit-config.yaml
```

Create `__init__.py` in every `src/` subdirectory and `tests/` subdirectory.

---

### Step 3 — `.gitignore`

```gitignore
# Data and models
data/raw/
data/processed/
models/
*.pkl
*.h5
*.onnx
*.tflite

# Environment
.env
.env.*
venv/
.venv/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/

# Notebooks
.ipynb_checkpoints/

# GCP
gcp-credentials.json
service-account*.json

# IDE
.vscode/settings.json
.idea/

# Logs
logs/
*.log

# OS
.DS_Store
```

---

### Step 4 — `requirements.txt`

```txt
tensorflow==2.15.0
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-multipart==0.0.7
Pillow==10.2.0
numpy==1.26.3
google-cloud-firestore==2.14.0
google-cloud-logging==3.9.0
onnx==1.15.0
onnxruntime==1.17.0
tf2onnx==1.16.1
optuna==3.5.0
click==8.1.7
```

---

### Step 5 — `requirements-dev.txt`

```txt
-r requirements.txt
pytest==7.4.4
pytest-cov==4.1.0
httpx==0.26.0
flake8==7.0.0
black==24.1.1
isort==5.13.2
pylint==3.0.3
pre-commit==3.6.0
locust==2.23.1
```

---

### Step 6 — `pyproject.toml`

```toml
[tool.black]
line-length = 88
target-version = ["py310", "py311"]

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=src --cov-report=term-missing"

[tool.pylint.messages_control]
disable = ["C0111", "R0903"]

[tool.pylint.format]
max-line-length = 88
```

---

### Step 7 — `.flake8`

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude =
    .git,
    __pycache__,
    venv,
    .venv,
    build,
    dist
```

---

### Step 8 — `Makefile`

```makefile
.PHONY: install lint format test clean

install:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	flake8 src/ tests/
	pylint src/ --fail-under=8.0

format:
	black src/ tests/
	isort src/ tests/

test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
```

---

### Step 9 — `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black

  - repo: https://github.com/pycco-docs/pycco
    rev: v0.6.0
    hooks:
      - id: isort
        name: isort

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
```

---

### Step 10 — Write one real test so pytest has something to run

`tests/unit/test_placeholder.py`:

```python
"""Placeholder tests — replaced as modules are built."""


def test_project_structure() -> None:
    """Verify the project can be imported."""
    assert True


def test_placeholder() -> None:
    """Remove this when real tests exist."""
    assert 1 + 1 == 2
```

---

### Step 11 — Initial commit

```bash

git init
git remote add origin https://github.com/Nasis-Salvatore-ML-Dev/aerospace-defect-detection.git


git add .
git commit -m "feat: initial project scaffolding

- Makefile with lint, format, test, install targets
- pytest structure with unit/integration/load directories
- flake8, black, isort, pylint configuration
- pre-commit hooks
- requirements.txt and requirements-dev.txt
- .gitignore for ML projects (data, models, credentials)
- pyproject.toml with tool configuration"

git push origin main
```

---

### Verification checklist before you come back

- [ ] `make install` runs without errors
- [ ] `make lint` runs (will pass on empty src/ — that is fine)
- [ ] `make test` runs and shows 2 tests passing
- [ ] `make format` runs without errors
- [ ] Repo is visible on GitHub with the full structure
- [ ] Commit message follows the conventional commits format above
