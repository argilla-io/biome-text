#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

if __name__ == "__main__":

    setup(
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
            "allennlp>=0.8.0",
            "torch>=1.0.0",
            "dask[complete]>=1.0,<2.0",
            # interactive console input
            "inquirer>=2.5.1,<2.6.0",
            "smart-open>=1.7.0",
            "coloredlogs",
            # github complains that pyyaml <4.0 is a security risk. 
            # We will use the newest one, some packages, though, will complain because they have not updated their requirements!
            "PyYAML<=3.13,>=3.10",
            "ujson",
            "spacy",
            "pandas",
            "elasticsearch>=6.0,<7.0",
            "bokeh",
            "xlrd>=1.0,<2.0"
        ],
        extras_require={"testing": ["pytest", "pytest-cov", "pytest-pylint"]},
        package_data={"biome": ["allennlp/commands/explore/ui/classifier.tar.gz"]},
        entry_points={"console_scripts": ["biome=biome.__main__:main"]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
