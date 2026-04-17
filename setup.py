from setuptools import find_packages, setup


setup(
    name="ml4sci-e2e",
    version="0.1.0",
    description="Dense baseline utilities for ML4SCI end-to-end particle-classification notebooks.",
    author="Shaik Muzamil",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "h5py>=3.10",
        "matplotlib>=3.8",
        "nbclient>=0.10",
        "nbformat>=5.10",
        "numpy>=2.0",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "seaborn>=0.13",
        "torch>=2.2",
        "torchvision>=0.17",
        "wheel>=0.45",
    ],
)
