import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="mitfat",
    version="0.2.0",
    url="https://github.com/vahid-sb/MiTfAT",
    license="GNU Version 3",
    author="Vahid Samadi Bokharaie",
    author_email="vahid.bokharaie@protonmail.com",
    description="A Python-based Scikit-Learn-friendly fMRI Analysis Tool, Made in Tuebingen.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests", "venv")),
    test_suite="nose.collector",
    tests_require=["nose"],
    package_data={
	},

    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "nibabel",
        "nilearn",
        "pathlib",
	"seaborn",
	"openpyxl",
    ],

    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
	"Programming Language :: Python :: 3.8",
    ],
)
