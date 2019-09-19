#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple
from setuptools import setup, find_packages
from pip import __version__ as pip_version

PIP_VERSION_REQUIRED = "19.1.1"


def check_pip_version(required_version: str, version: str):
    def version_str_2_numbers(version: str) -> Tuple[(int, int, int)]:
        return tuple([int(n) for n in version.split(".")])

    mayor, minor, fixes = version_str_2_numbers(version)
    req_mayor, req_minor, req_fixes = version_str_2_numbers(required_version)

    if mayor >= req_mayor and minor >= req_minor and fixes >= req_fixes:
        pass
    else:
        print(f"Minimal pip version should be {required_version}, found: {version}")
        print(f"Please upgrade pip: pip install --upgrade pip")
        exit(1)


if __name__ == "__main__":
    check_pip_version(PIP_VERSION_REQUIRED, pip_version)

    setup(
        version="0.1.dev",
        name="biome-text",
        description="Biome-text is a light-weight open source Natural Language Processing toolbox"
        "tool built with AllenNLP",
        author="Recognai",
        author_email="francisco@recogn.ai",
        url="https://www.recogn.ai/",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=find_packages("src"),
        package_dir={"": "src"},
        install_requires=[
            "urllib3>=1.21.1,<1.25",  # avoids version conflicts
            "requests<=2.21.0",  # avoid version conflicts
            "allennlp~=0.8.0",
            "cachey",  # Needed to use the Cache class in dask
            "dask[complete]~=2.0",
            "pyarrow~=0.13",
            # interactive console input
            "inquirer>=2.5.1,<2.6.0",
            "smart-open>=1.7.0",
            "coloredlogs~=10.0",
            "PyYAML>=3.10,<=5.1",  # github complains that pyyaml <4.0 is a security risk!
            "ujson~=1.35",
            # "spacy~=2.1.0",
            "pandas~=0.24.0",
            "elasticsearch>=6.0",
            "bokeh~=1.2.0",
            "xlrd>=1.0,<2.0",
            "flatdict~=3.2.0",
        ],
        extras_require={"testing": ["pytest", "pytest-cov", "pytest-pylint"]},
        package_data={"biome": ["allennlp/commands/explore/ui/classifier.tar.gz"]},
        entry_points={"console_scripts": ["biome=biome.__main__:main"]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
