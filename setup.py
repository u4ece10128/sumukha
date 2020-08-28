# -*- coding: utf-8 -*-

from configparser import ConfigParser

from setuptools import find_packages, setup

config = ConfigParser()
config.read('setup.cfg')
version = config.get('project', 'version')

setup(
    name='sumukha',
    version=version,
    description='Jays SPI',
    author='Jaya Ram Kollipara',
    author_email='jayaram.kollipara@igenius.ai',
    license='Apache License 2.0',
    url='https://github.com/u4ece10128/sumukha',
    py_modules=['version'],
    packages=find_packages(where='src', exclude=('tests.*', 'tests')),
    package_dir={'': 'src'},
    zip_safe=False,
    install_requires=[
        'numpy==1.18.5',
        'pandas==1.0.3',
        'dotmap==1.2.17',
        'fasttext==0.9.2',
        'igenius.brushes[full]==18.1.1',
        'bs4==0.0.1',
        'scikit-learn==0.20.3',
        ],
    extras_require={
        'dev': [
            'psycopg2-binary==2.7.6.1',
            'PyYAML==5.3',
            'flake8==3.3.0',
            'flake8-builtins==1.2.3',
            'flake8-commas==2.0.0',
            'flake8-comprehensions==1.4.1',
            'flake8-debugger==3.1.0',
            'flake8-import-order==0.17.1',
            'flake8-quotes==0.14.1',
            'flake8-todo==0.7',
            'ipdb==0.12.3',
            'pep8-naming==0.5.0',
            'bumpversion==0.5.3',
            'jupyter==1.0.0',
        ],
        'tf': [
            'tensorflow==2.2.0',
        ]
    },
)
