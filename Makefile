.PHONY: default test dist install dev specs
default: help



bin_dir = tools/bin
swagger-codegen = java -jar $(bin_dir)/swagger-codegen-cli.jar
swagger-codegen-version ?= 2.4.0
maven-repo ?= http://central.maven.org/maven2

swagger-codegen-url=$(maven-repo)/io/swagger/swagger-codegen-cli/$(swagger-codegen-version)/swagger-codegen-cli-$(swagger-codegen-version).jar

test: ## launch package tests
	@python setup.py test

dist: ## build a package distribution with tests
	@python setup.py test bdist_wheel

install: ## install package
	@python setup.py install

dev: ## install package in develop mode
	@python setup.py develop

submodules: ## sync git submodules in repo
	@git submodule init && exit
	@git submodule sync --recursive && git submodule update --recursive --remote
	@cp -R modules/biome-data/src/biome/data biome/

# Do not use: generate data model from api spec
specs:
	@wget $(swagger-codegen-url) -O $(bin_dir)/swagger-codegen-cli.jar
	@$(swagger-codegen) generate \
	-i ~/recognai/biome/biome-api/spec.yml \
	-o . \
    -l python \
    -DpackageName=biome \
    --model-package spec \
    --api-package api \
    --ignore-file-override ./.swagger-codegen-ignore


.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'