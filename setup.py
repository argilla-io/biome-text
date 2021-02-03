#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob

from pip import __version__ as pip_version
from pkg_resources import parse_version
from setuptools import setup

try:
    from setuptools import find_namespace_packages
except ImportError as error:
    raise ImportError("Make sure you have setuptools >= 40.1.0 installed!") from error

# We require a fairly new version to make use of the new dependency resolver!
REQUIRED_PIP_VERSION = "20.3.0"


if __name__ == "__main__":
    if parse_version(pip_version) <= parse_version(REQUIRED_PIP_VERSION):
        raise OSError(
            f"Minimal required pip version is {REQUIRED_PIP_VERSION}, found: {pip_version}\n"
            "Please upgrade pip: pip install --upgrade pip"
        )

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
            "allennlp~=1.4.0",
            "beautifulsoup4~=4.9.0",
            "captum~=0.2.0",
            "click~=7.1.0",
            "datasets~=1.2.1",
            "elasticsearch>=6.8.0,<7.5.0",
            "fastapi~=0.55.0",
            "flask~=1.1.2",
            "flask-cors~=3.0.8",
            "flatdict~=4.0.0",
            "gevent~=20.9.0",
            "ipywidgets~=7.5.1",
            "lxml~=4.6.2",
            "mlflow~=1.13.1",
            "pandas~=1.1.0",
            "ray[tune]~=1.0.0",
            "spacy~=2.3.0",
            "tqdm>=4.49.0",
            "uvicorn~=0.11.0",
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
                "GitPython",
            ],
        },
        package_data={
            "biome": [
                file.replace("src/biome/", "")
                for file in glob.glob("src/biome/text/ui/webapp/**/*.*", recursive=True)
            ]
        },
        entry_points={"console_scripts": ["biome=biome.text.cli:main"]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
