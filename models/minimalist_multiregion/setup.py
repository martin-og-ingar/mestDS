# This file is used to install the package in the system.
# Evaluate the need for a setup.py file for the project later on.

from setuptools import setup, find_packages

requirements = [
    "numpy",
    "matplotlib",
    "chap-core",
]

setup(
    name="mestDS",
    version="0.1",
    packages=find_packages(where="src"),
    install_requires=requirements,
    package_dir={"": "src"},
)
