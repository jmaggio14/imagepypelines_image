#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='imagepypelines_image',
      packages=find_packages(),
      entry_points={'imagepypelines.plugins': 'image = imagepypelines_image'},
      include_package_data=True,
      )
