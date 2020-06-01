#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import sys
from typing import Tuple

try:
    from setuptools import setup, find_namespace_packages
except ImportError as error:
    raise ImportError("Make sure you have setuptools >= 40.1.0 installed!") from error

from pip import __version__ as pip_version

PIP_VERSION_REQUIRED = "19.1.1"


def check_pip_version(required_version: str, version: str) -> bool:
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
        return True

    print(f"Minimal pip version should be {required_version}, found: {version}")
    print(f"Please upgrade pip: pip install --upgrade pip")
    return False


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
    if not check_pip_version(PIP_VERSION_REQUIRED, pip_version):
        sys.exit(1)

    package_name = "biome-text"
    about = about_info(package_name)

    setup(
        name=package_name,
        version=about["__version__"],
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
            "allennlp~=1.0.0rc",
            "gevent~=1.4.0",
            "flask~=1.1.2",
            "flask-cors~=3.0.8",
            "click~=7.1.0",
            "beautifulsoup4~=4.9.0",
            "lxml~=4.5.0",
            "fastapi~=0.55.0",
            "uvicorn~=0.11.0",
            "dask[complete]~=2.17.0",
            "msgpack~=0.6.0",
            "cachey~=0.2.0",
            "pyarrow~=0.17.0",
            "ujson~=2.0.0",
            "pandas~=1.0.0",
            "dask-elk~=0.4.0",
            "bokeh~=2.0.0",
            "xlrd~=1.2.0",
            "flatdict~=4.0.0",
            "s3fs~=0.4.0",
            "captum~=0.2.0",
            "ipywidgets~=7.5.1"
        ],
        extras_require={
            "testing": [
                "pytest",
                "pytest-cov",
                "pytest-pylint~=0.14.0",
                "black",
                "GitPython",
                "pdoc3~=0.8.1",
                "pytest-notebook~=0.6.0",
            ]
        },
        package_data={
            "biome": [
                file.replace("src/biome/", "")
                for file in glob.glob("src/biome/text/ui/webapp/**/*.*", recursive=True)
            ]
        },
        entry_points={"console_scripts": ["biome=biome.text.cli:main",]},
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )
