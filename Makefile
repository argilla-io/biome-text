.PHONY: default test dist install dev pip-install-dev docs ui
default: help

check: ## applies a code pylint with autopep8 reformating
	@pre-commit run --all-files
	@pylint --exit-zero --rcfile=setup.cfg --unsafe-load-any-extension=y src

test: check ## launch package tests
	@pytest

dist: test ui ## run tests and build a package distribution
	@python setup.py sdist bdist_wheel

install: ## install package
	@pip install .

dev: ## install package in development mode
	@pip install --use-feature=2020-resolver --upgrade -e .[testing]
	@pre-commit install

ui: ## build the ui pages
	@cd ui && npm install && npm run build

docs: ## build the documentation site
	@cd docs && npm install && npm run build:site

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
