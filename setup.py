
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='BlueBird',
    version='0.0.1',
    description='Deep learning library',
    long_description=readme,
    author='Gordan Prastalo',
    author_email='gordan.prastalo.gp@gmail.com',
    url='https://github.com/...',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)