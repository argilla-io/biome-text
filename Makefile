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

.PHONY: dist
dist: test
	@python setup.py bdist_wheel

install: dist
	@pip install dist/*.whl

dev:
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