from setuptools import setup, find_packages
from gimg_utils import __version__


setup(
    name="gimg_utils",
    version=__version__,
    description="Some tools to quickly read geographical images and iterate over the image content by tiles",
    author="vfdev-5",
    author_email="vfdev.5@gmail.com",
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=[
        'numpy', 
    ],
    test_suite="tests",
    extras_require={
          'tests': ['pytest',]
    }
)
