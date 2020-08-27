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
    name="MiTfAT",
    version="0.1.6",
    url="https://gitlab.tuebingen.mpg.de/vbokharaie/mitfat/",
    license="GNU Version 3",
    author="Vahid Samadi Bokharaie",
    author_email="vahid.bokharaie@tuebingen.mpg.de",
    description="A python-based fMRI Analysis Tool. ",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests", "venv")),
    test_suite="nose.collector",
    tests_require=["nose"],
    package_data={"mitfat": ["datasets/*.*", "sample_info_file.txt"]},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "nibabel",
        "nilearn",
        "pathlib",
	"click",
	"seaborn",
	"openpyxl",
    ],

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
)
