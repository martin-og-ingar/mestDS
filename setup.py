# This file is used to install the package in the system.
# Evaluate the need for a setup.py file for the project later on.

from setuptools import setup, find_packages

setup(
    name="mestDS",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)