# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='allennlp_extensions',
    version='0.0.1',
    description='Allen nlp custom extensions',
    long_description=readme,
    author='recognai',
    author_email='contact@recogn.ai',
    url='https://bitbucket.org/recognai/allennlp-extensions',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

