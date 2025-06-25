#!/usr/bin/env python

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

packages = find_packages()
install_requires = [line.strip() for line in open('requirements.txt')]

setup(name='rtac',
      version='0.0.1.2',
      description=("RTAC is a collection of Realtime Algorithm" 
                   " Configuration methods."),
      long_description=README,
      long_description_content_type="text/markdown",
      url='https://github.com/DimitriWeiss/RTAC.git',
      author='Dimitri Weiß',
      author_email='dimitri-weiss@web.de',
      packages=find_packages(exclude=["*.tests"]),
      include_package_data=True,
      license='LICENSE',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: POSIX :: Linux',        
          'Programming Language :: Python :: 3.10',
      ],
      install_requires=install_requires,
      entry_points={},
      )
