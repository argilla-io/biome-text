.PHONY: default dev check test ui build_ui docs build_docs dist
default: help

dev: ## install package in development mode
	@pip install --upgrade -e .[dev]
	@pre-commit install

check: ## applies a code pylint with autopep8 reformating
	@pre-commit run --all-files
	@pylint --exit-zero --rcfile=setup.cfg --unsafe-load-any-extension=y src

test: ## launch package tests
	@python -m pytest

ui: ## serve the UI for development
	@cd ui && npm install && npm run serve

build_ui: ## build the ui pages
	@cd ui && npm install && npm run build

docs: ## serve the documentation for development
	@cd docs && npm install && npm run dev:site

build_docs: ## build the documentation
	@cd docs && npm install && npm run build:site

dist: build_ui ## build a package distribution
	@python setup.py sdist bdist_wheel


.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
