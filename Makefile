.PHONY: default test dist install dev
default: help

test: ## launch package tests
	@python setup.py test

dist: ## build a package distribution with tests
	@python setup.py test bdist_wheel

install: ## install package
	@python setup.py install

dev: ## install package in develop mode
	@python setup.py develop

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'