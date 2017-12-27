init:
	@pip install -r requirements.txt

spacy:
	@python -m spacy download en_core_web_sm

init-test:
	@pip install -r requirements_test.txt
	
test: init-test
	@nosetests tests


