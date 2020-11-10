#!/usr/bin/env python
from setuptools import setup, find_packages
import os

# REPLACE THESE VARIABLES
# this should be the name of your source directory like "imagepypelines_template"
DIRECTORY_NAME = "imagepypelines_image"
# this should be the name your namespace - so your plugin will be available as
# imagepypelines.<namespace>
NAMESPACE = "image"


# ==============================================================================
#               YOU DON'T HAVE TO MODIFY ANYTHING BELOW HERE
# ==============================================================================
current_dir = os.path.dirname(__file__)

# load __version__, __author__, __email__, etc variables
with open( os.path.join(current_dir, DIRECTORY_NAME, 'globals.py') ) as f:
    exec(f.read())

# loads requirements from requirements.txt file
requirements = ''
with open( os.path.join(current_dir, 'requirements.txt'), 'r' ) as f:
    requirements = f.readlines()

# load in list of opencv requirements
cv_requirements_path = os.path.join(current_dir,'requirements-cv.txt')
with open(cv_requirements_path,'r') as f:
    requirements_cv = f.read().splitlines()

# fetches the readme text
readme_text = ''
with open( os.path.join(current_dir, 'README.rst'), 'r' ) as f:
    readme_text = f.read()

setup(name=DIRECTORY_NAME,
      packages=find_packages(),
      entry_points={'imagepypelines.plugins': f'{NAMESPACE} = {DIRECTORY_NAME}'},
      include_package_data=True,
      install_requires=requirements,
      long_description=readme_text,
      long_description_content_type='text/x-rst',
      # from globals.py
      version=__version__,
      description=__description__,
      author=__author__,
      author_email=__email__,
      license=__license__,
      extras_require = {
                    'cv' : requirements_cv,
                    },
      )
