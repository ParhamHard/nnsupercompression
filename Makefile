.PHONY: help install install-dev test test-cov lint format clean build docs

help:  ## Show this help message
	@echo "Neural Network Supercompression Library"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	python -m pytest tests/

test-cov:  ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 src/ tests/
	mypy src/

format:  ## Format code with black
	black src/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python setup.py sdist bdist_wheel

docs:  ## Build documentation
	cd docs && make html

run-basic:  ## Run basic compression example
	python examples/basic_compression.py

run-perfect:  ## Run perfect compression example
	python examples/perfect_compression.py

run-architectures:  ## Run architecture comparison example
	python examples/architecture_experiments.py

run-demo:  ## Run structure demo
	python examples/demo_structure.py

examples: run-basic run-perfect run-architectures run-demo  ## Run all examples

check: lint test  ## Run linting and tests

all: clean install-dev check examples  ## Run full development workflow
