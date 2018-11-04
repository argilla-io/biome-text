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

init-test:
	@pip install -r requirements.test.txt

test: init-test check
	@nosetests --with-coverage --cover-package=biome -d tests

.PHONY: check
check:
	@pylint -E biome
	@echo "lint succeeded"

.PHONY: dist
dist: test
	@python setup.py sdist

install: test
	@python setup.py install

install-dev:
	@python setup.py develop
