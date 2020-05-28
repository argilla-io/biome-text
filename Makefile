.PHONY: default test dist install dev pip-install-dev docs ui
default: help

check: ## applies a code pylint with autopep8 reformating
	@black .
	@pylint --exit-zero --rcfile=setup.cfg --unsafe-load-any-extension=y src

test: check ## launch package tests
	@pytest

dist: test ui ## run tests and build a package distribution
	@python setup.py sdist bdist_wheel

install: ## install package
	@pip install .

pip-install-dev:
	@pip install --upgrade -e .[testing]

dev: pip-install-dev ui ## install package in development mode

ui: ## build the ui pages
	@cd ui && npm install && npm run build

docs: ## build the documentation site
	@cd docs && npm install && npm run build:site

# TODO: remove it
upgrade-classifier-ui: ## updates the biome-classifier-ui interface artifact
	@curl \
    --output src/biome/text/commands/ui/classifier.tar.gz \
    -L \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    ${GITLAB_BIOME_CLASSIFIER_UI_ARTIFACT_URL}

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
