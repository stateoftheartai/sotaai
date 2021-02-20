'''
Core sotaai library setup file
'''

from setuptools import setup

setup(name='sotaai',
      version='0.0.4',
      author='Stateoftheart AI PBC',
      author_email='admin@stateoftheart.ai',
      description='Stateoftheart AI Official Library',
      url='https://github.com/stateoftheartai/sotaai',
      packages=['sotaai', 'sotaai.cv'],
      install_requires=[
          'tensorflow', 'tensorflow-datasets', 'fastai==1.0.61', 'gym'
      ],
      python_requires='>=3,<3.9',
      extras_require={})
