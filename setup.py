#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for biome_allennlp.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import require, VersionConflict
from setuptools import setup

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        long_description_content_type='text/markdown',
        package_data={'biome': [
            'allennlp/commands/explore/ui/**/*.*',
            'allennlp/commands/explore/ui/**/**/*.*',
            'allennlp/commands/explore/ui/**/**/**/*.*',
        ]}
    )
