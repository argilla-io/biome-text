
init:
	@pip install -r requirements.install.txt --upgrade

spacy-es:
	@python -m spacy download es_core_news_sm

spacy:
	@python -m spacy download en_core_web_sm

init-test:
	@pip install -r requirements.test.txt
	
test: init-test
	@nosetests --with-coverage --cover-package=allennlp_extensions -d tests

install:
	@python setup.py install
install-dev:
	@python setup.py develop
