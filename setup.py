"""
Setup script for Neural Network Supercompression Library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nnsupercompression",
    version="1.0.0",
    author="Neural Compression Team",
    author_email="team@neuralcompression.com",
    description="A library for compressing data using tiny neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nnsupercompression",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nnsupercompression/issues",
        "Source": "https://github.com/yourusername/nnsupercompression",
        "Documentation": "https://github.com/yourusername/nnsupercompression#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Archiving :: Compression",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nnsupercompression=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="neural-network, compression, autoencoder, machine-learning, deep-learning",
)
