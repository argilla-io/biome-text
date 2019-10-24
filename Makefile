.PHONY: default test dist install dev
default: help

test: ## launch package tests
	@pytest

dist: ## run tests and build a package distribution
	@pytest && python setup.py sdist bdist_wheel

install: ## install package
	@pip install .

dev: ## install package in development mode
	@pip install -U git+https://github.com/recognai/biome-data.git
	@pip install --upgrade -e .[testing]

upgrade-classifier-ui: ## updates the biome-classifier-ui interface artifact
	@curl \
    --output src/biome/text/commands/ui/classifier.tar.gz \
    -L \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    ${GITLAB_BIOME_CLASSIFIER_UI_ARTIFACT_URL}

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
