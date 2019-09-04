#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

if __name__ == "__main__":

    setup(
        version="0.1.dev",
        name="biome-text",
        description="Biome-text is a light-weight open source Natural Language Processing "
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
            "jsonschema~=2.6",  # avoids version conflicts
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
            "spacy~=2.1.0",
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
