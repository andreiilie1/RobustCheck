from setuptools import setup, find_packages

requirements = [
    "matplotlib >= 3.7.0",
    "numpy >= 1.24.2",
    "mlflow >= 1.2.0",
]

setup(
    name="robustcheck",
    description="Tool for quick assessment of image classifiers robustness",
    url="",
    author="Andrei Ilie",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=requirements,
)
