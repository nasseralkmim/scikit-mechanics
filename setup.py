try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='scikit-mechanics',
      version='0.1-dev',
      description='Suit for solving mechanics problems',
      long_description=open('README.rst').read(),
      license='GPLv3',
      author='The scikit-mechanics contributors',
      packages=['skmech'],
      install_requires=["numpy", "matplotlib"]
      )
