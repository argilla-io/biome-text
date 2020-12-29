#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os

try:
    from setuptools import find_namespace_packages
    from setuptools import setup
except ImportError as error:
    raise ImportError("Make sure you have setuptools >= 40.1.0 installed!") from error

from pip import __version__ as pip_version

# We require a fairly new version to make use of the new dependency resolver!
REQUIRED_PIP_VERSION = "20.3.0"


def check_pip_version():
    def version2int(version: str) -> int:
        version_fractions = [f"{int(n):02d}" for n in version.split(".")]
        return int("".join(version_fractions))

    if version2int(pip_version) < version2int(REQUIRED_PIP_VERSION):
        raise OSError(
            f"Minimal required pip version is {REQUIRED_PIP_VERSION}, found: {pip_version}\n"
            "Please upgrade pip: pip install --upgrade pip"
        )


if __name__ == "__main__":
    check_pip_version()

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
            "allennlp~=1.3.0",
            "spacy~=2.3.0",
            "gevent~=20.9.0",
            "flask~=1.1.2",
            "flask-cors~=3.0.8",
            "click~=7.1.0",
            "beautifulsoup4~=4.9.0",
            "lxml~=4.5.0",
            "fastapi~=0.55.0",
            "uvicorn~=0.11.0",
            "distributed~=2.17.0",
            "cachey~=0.2.0",
            "pandas~=1.1.0",
            "xlrd~=1.2.0",
            "flatdict~=4.0.0",
            "s3fs~=0.4.0",
            "captum~=0.2.0",
            "ipywidgets~=7.5.1",
            "mlflow~=1.9.0",
            "elasticsearch>=6.8.0,<7.5.0",
            "ray[tune]~=1.0.0",
            "datasets~=1.1.2",
            "tqdm>=4.49.0",
        ],
        extras_require={
            "dev": [
                # testing
                "pytest>=6.2.0",
                "pytest-cov>=2.10.0",
                "pytest-pylint>=0.14.0",
                "pytest-notebook~=0.6.0",
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
