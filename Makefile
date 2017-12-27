init:
	@pip install -r requirements.install.txt

spacy:
	@python -m spacy download en_core_web_sm

init-test:
	@pip install -r requirements.test.txt
	
test: init-test
	@nosetests tests

install:
	@python setup.py install


