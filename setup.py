#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from pip import __version__ as pip_version
from setuptools import setup, find_packages
from typing import Tuple

PIP_VERSION_REQUIRED = "19.1.1"


def check_pip_version(required_version: str, version: str):
    def version_str_2_numbers(version: str) -> Tuple[int, int, int]:
        version_fractions = [int(n) for n in version.split(".")]
        return tuple(
            [
                version_fractions[i] if i < len(version_fractions) else 0
                for i in range(0, 3)
            ]
        )

    mayor, minor, fixes = version_str_2_numbers(version)
    req_mayor, req_minor, req_fixes = version_str_2_numbers(required_version)

    if (
        mayor > req_mayor
        or (mayor == req_mayor and minor > req_minor)
        or (mayor == req_mayor and minor == req_minor and fixes >= fixes)
    ):
        pass
    else:
        print(f"Minimal pip version should be {required_version}, found: {version}")
        print(f"Please upgrade pip: pip install --upgrade pip")
        exit(1)


def about_info(package: str):
    """Fetch about info """
    root = os.path.abspath(os.path.dirname(__file__))
    with open(
        os.path.join(root, "src", package.replace("-", "/"), "about.py"),
        encoding="utf8",
    ) as f:
        about = {}
        exec(f.read(), about)
        return about


if __name__ == "__main__":
    check_pip_version(PIP_VERSION_REQUIRED, pip_version)

    package_name = "biome-text"
    about = about_info(package_name)

    setup(
        name=package_name,
        version=about["__version__"],
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
            "allennlp~=0.9",
            "smart_open~=1.8",
            "coloredlogs==10.0",
            "dask-elk~=0.2.0",
            "elasticsearch<7.0",  # latest version doesn't work with dask-elk module
            "biome-data~=0.2.0",
        ],
        extras_require={
            "testing": ["pytest", "pytest-cov", "pytest-pylint~=0.14.0", "black", "GitPython"]
        },
        package_data={"biome": ["text/commands/ui/classifier.tar.gz"]},
        entry_points={"console_scripts": ["biome=biome.text.__main__:main"]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
