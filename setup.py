from setuptools import setup
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(name='mpot',
      description="Implementation of MPOT in PyTorch",
      author="An T. Le",
      author_email="an@robot-learning.de",
      packages=['mpot'],
      install_requires=requires_list,
)
