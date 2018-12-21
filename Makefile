.PHONY: test dist install dev specs

bin_dir = tools/bin
swagger-codegen = java -jar $(bin_dir)/swagger-codegen-cli.jar
swagger-codegen-version ?= 2.4.0
maven-repo ?= http://central.maven.org/maven2

swagger-codegen-url=$(maven-repo)/io/swagger/swagger-codegen-cli/$(swagger-codegen-version)/swagger-codegen-cli-$(swagger-codegen-version).jar

test:
	@python setup.py test

dist:
	@python setup.py test bdist_wheel

install:
	@pip install dist/*.whl

dev:
	@python setup.py develop




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
