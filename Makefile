.PHONY: install lint format test clean

install:
	pip install uv
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt
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
