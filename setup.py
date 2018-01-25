from codecs import open as codecs_open
from setuptools import setup, find_packages
from gimg import __version__


# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="gimg",
    version=__version__,
    description=u"Tiny package with purpose of reading and manipulation of geographical images using GDAL",
    long_description=long_description,
    author="vfdev-5",
    author_email="vfdev dot 5 at gmail dot com",
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=[
        'numpy',
        'click',
    ],
    license='MIT',
    test_suite="tests",
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov'
        ]
    },
    entry_points="""
        [console_scripts]
            tile_generator=gimg.cli.tile_generator:cli
    """
)
