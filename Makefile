NAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	platform=linux-amd64
endif
ifeq ($(UNAME_S),Darwin)
	platform=darwin-amd64
endif
init:
	@pip install -r requirements.txt --upgrade
	@pip install pylint

spacy-es:
	@python -m spacy download es_core_news_sm
	@python -m spacy link es_core_news_sm es -f

spacy:
	@python -m spacy download en_core_web_sm


test:
	@python setup.py test

.PHONY: check
check:
	@pylint -E biome
	@echo "lint succeeded"

.PHONY: dist
dist: test
	@python setup.py sdist bdist_wheel

install: dist
	@pip install dist/*.whl

install-dev:
	@python setup.py develop

generate-specs:
	@swagger-codegen generate \
	-i ~/recognai/biome/apis/engine-api/api.yaml \
	-o . \
    -l python \
    -DpackageName=biome \
    --model-package spec \
    --api-package api \
    --ignore-file-override ./.swagger-codegen-ignore