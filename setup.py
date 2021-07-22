#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob

from setuptools import setup

try:
    from setuptools import find_namespace_packages
except ImportError as error:
    raise ImportError("Make sure you have setuptools >= 40.1.0 installed!") from error


if __name__ == "__main__":
    setup(
        name="biome-text",
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        description="Biome-text is a light-weight open source Natural Language Processing toolbox"
        " built with AllenNLP",
        author="Recognai",
        author_email="francisco@recogn.ai",
        url="https://www.recogn.ai/",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=find_namespace_packages("src"),
        package_dir={"": "src"},
        install_requires=[
            "allennlp~=2.5.0",
            "beautifulsoup4~=4.9.0",
            "captum~=0.2.0",
            "click~=7.1.0",
            "datasets~=1.10.0",
            "flatdict~=4.0.0",
            "ipywidgets~=7.5.1",
            "lxml~=4.6.2",
            "mlflow~=1.13.1",
            "numpy",
            "pandas<=1.3.0",
            "pytorch-lightning~=1.3.0",
            "ray[tune]~=1.3.0",
            "spacy~=3.0.0",
            "torch",  # the version is defined by allennlp
            "transformers",  # the version is defined by allennlp
            "tqdm>=4.49.0",
            "fastapi~=0.63.0",  # newer versions brings pydantic conflicts with spaCy 3.0.x
            "uvicorn>=0.13.0",
            "pyyaml",
        ],
        extras_require={
            "dev": [
                # testing
                "pytest>=6.2.0",
                "pytest-cov>=2.10.0",
                "pytest-pylint>=0.14.0",
                "pytest-notebook~=0.6.0",
                "wandb>=0.10.12",
                "xlrd~=1.2.0",
                # documentation
                "pdoc3~=0.8.1",
                # development
                "pre-commit~=2.9.0",
                # "GitPython",
            ],
        },
        entry_points={"console_scripts": ["biome=biome.text.cli:main"]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
