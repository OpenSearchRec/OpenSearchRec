from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('models_requirements.txt') as f:
    models_requirements = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

setup(
    name='OpenSearchRec',
    version='0.2.0',
    author='Sebastien L',
    packages=find_packages(
        include="OpenSearchRec*"
    ),
    license='Apache 2',
    description='OpenSearchRec: Open Source Search and Recommendations',
    long_description=readme,
    install_requires=required,
    extras_require={
        "models": models_requirements
    }
)
