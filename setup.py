
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='bluebird-stoick01',
    version='0.0.4',
    description='Deep learning library',
    long_description_content_type='text/markdown',
    long_description=readme,
    author='Gordan Prastalo',
    author_email='gordan.prastalo.gp@gmail.com',
    url='https://github.com/Stoick01/bluebird',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires='>=3.6',
    test_suite="tests",
)