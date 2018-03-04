
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
	@nosetests --with-coverage --cover-package=allennlp_extensions -d tests

.PHONY: build
build:
	@pylint allennlp_extensions -E

install: build
	@python setup.py install

install-dev: build
	@python setup.py develop
