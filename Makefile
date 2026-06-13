.PHONY: setup lint test run clean

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pre-commit install

lint:
	black src/ tests/
	flake8 src/ tests/
	isort src/ tests/

test:
	pytest tests/ -v

run:
	python pipeline/run_pipeline.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
