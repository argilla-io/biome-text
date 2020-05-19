#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
            "allennlp~=0.9.0",
            "click~=7.0.0",
            "smart_open~=1.8.0",
            "coloredlogs~=10.0.0",
            "elasticsearch<7.0",  # latest version doesn't work with dask-elk module
            "beautifulsoup4~=4.8.2",
            "lxml~=4.5.0",
            "fastapi~=0.52.0",
            "uvicorn~=0.11.0",
            "dask[complete]~=2.10.0",
            "msgpack~=0.6.0",
            "cachey~=0.1.0",  # required by dask.cache
            "pyarrow~=0.15.0",
            "ujson~=1.35",
            "pandas~=0.25.0",
            "elasticsearch<7.0",  # latest version doesn't work with dask-elk module
            "dask-elk~=0.3.0",
            "bokeh~=1.3.0",
            "xlrd~=1.2.0",
            "flatdict~=3.4.0",
            "python-dateutil<2.8.1",  # botocore (imported from allennlp) has this restriction
            "s3fs~=0.4.0",
            "captum~=0.2.0",
        ],
        extras_require={
            "testing": [
                "pytest",
                "pytest-cov",
                "pytest-pylint~=0.14.0",
                "black",
                "GitPython",
                "pdoc3~=0.8.1",
            ]
        },
        package_data={"biome": ["text/commands/ui/classifier.tar.gz"]},
        entry_points={
            "console_scripts": [
                "biome=biome.text.__main__:main",
                "biome-new=biome.text.api_new.cli:main",
            ]
        },
        python_requires=">=3.6.1",  # taken from AllenNLP
        zip_safe=False,
    )