from setuptools import setup, find_packages

setup(
    name='OpenSearchRec',
    version='0.1.2',
    author='Sebastien L',
    packages=find_packages(
        include="OpenSearchRec*"
    ),
    license='Apache 2',
    description='OpenSearchRec: Open Source Search and Recommendations',
    long_description='OpenSearchRec: Open Source Search and Recommendations. Code: https://github.com/OpenSearchRec/OpenSearchRec',
    install_requires=[
        "fastapi>=0.76.0",
        "uvicorn[standard]>=0.17.6",
        "pytest>=7.1.2",
        "requests>=2.28.0",
        "httpx>=0.23.0",
        "numpy>=1.23.2"
    ],
    extras_require={
        "models": [
            "implicit",
            "sentence-transformers"
        ]
    }
)
