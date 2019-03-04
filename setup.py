#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

if __name__ == "__main__":

    setup(
        name="biome-allennlp",
        description="This package wraps and adds some new functionalities to "
        "the AllenNLP library. It is used by Biome to train and "
        "apply NLP models.Biome plaftform for deep learning "
        "models.",
        author="Recognai",
        author_email="francisco@recogn.ai",
        url="https://www.recogn.ai/",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=find_packages(
            where="./src",
            include=["biome.*"],
            exclude=["tests", "*.tests", "*.tests.*", "tests.*"],
        ),
        package_dir={"": "src"},
        install_requires=[
            "allennlp>=0.8.0",
            "torch>=1.0.0",
            "dask[complete]>=1.0,<2.0",
            # interactive console input
            "inquirer>=2.5.1,<2.6.0",
            "smart-open>=1.7.0",
            "coloredlogs",
            # fix pyyaml version to avoid package incompatibilities
            "pyyaml>=3.0,<4.0",
            # private repositories, CHANGE FOR OPENSOURCE RELEASE!
            "biome-data @ git+https://gitlab+deploy-token-48918:Lhgdcy6sa_9xBnLyaN7u@gitlab.com/recognai-team/biome/biome-data.git",
        ],
        extras_require={"testing": ["pytest", "pytest-cov", "pytest-pylint"]},
        entry_points={"console_scripts": ["biome=biome.__main__:main"]},
        include_package_data=True,
        package_data={"": ["LICENSE.txt", "src/biome/allennlp/commands/explore/ui/*"]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
