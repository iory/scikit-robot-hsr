#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup_requires = []
install_requires = [
    'scikit-robot',
]

setup(
    name='scikit-robot-hsr',
    version='0.0.1',
    description='A HSR Robot Interface Library in Python',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/scikit-robot-hsr',
    license='MIT License',
    packages=find_packages(),
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
