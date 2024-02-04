from setuptools import setup, find_packages

requirements = [
    "matplotlib >= 3.7.0",
    "numpy >= 1.18.0",
    "mlflow >= 1.2.0",
    "tqdm >= 4.64.1",
]

setup(
    name="robustcheck",
    packages=find_packages(),
    description="Tool for quick assessment of image classifiers robustness",
    url="https://github.com/andreiilie1/RobustCheck",
    download_url="https://github.com/andreiilie1/RobustCheck/archive/refs/tags/v1.0.8.tar.gz",
    author="Andrei Ilie",
    author_email="andrei0758@gmail.com",
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=requirements,
)
