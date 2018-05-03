NAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	platform=linux-amd64
endif
ifeq ($(UNAME_S),Darwin)
	platform=darwin-amd64
endif
init:
	@pip install -r requirements.install.txt --upgrade
	@pip install pylint

spacy-es:
	@python -m spacy download es_core_news_sm

spacy:
	@python -m spacy download en_core_web_sm

init-test:
	@pip install -r requirements.test.txt

test: init-test build
	@nosetests --with-coverage --cover-package=recognai -d tests

.PHONY: build
build: 
	@pylint recognai -E

install: test
	@python setup.py install

install-dev: build
	@python setup.py develop
