'''
Core sotaai library setup file
'''

from setuptools import setup

setup(name='sotaai',
      version='0.0.7',
      author='Stateoftheart AI PBC',
      author_email='admin@stateoftheart.ai',
      description='Stateoftheart AI Official Library',
      url='https://github.com/stateoftheartai/sotaai',
      packages=['sotaai', 'sotaai.cv'],
      install_requires=[],
      python_requires='>=3,<=3.8.0',
      extras_require={
          'cv': [
              'tensorflow', 'tensorflow-datasets', 'torchvision', 'torch',
              'transformers', 'scikit-image', 'opencv-python==4.5.1.48'
          ],
          'neuro': [],
          'nlp': ['datasets'],
          'rl': ['gym']
      })
