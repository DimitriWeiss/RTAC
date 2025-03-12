#!/usr/bin/env python

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(name='rtac',
      version='0.0.1.dev',
      long_description=README,
      long_description_content_type="text/markdown",
      description='Realtime Algorithm Configuration Methods',
      url='https://github.com/DimitriWeiss/RTAC.git',
      author='Dimitri Weiß',
      author_email='dimitri-weiss@web.de',
      packages=find_packages(exclude=["*.tests"]),
      include_package_data=True,
      license='LICENSE',
      )
