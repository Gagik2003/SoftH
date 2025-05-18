# setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='SoftH',
    version='0.1.0',
    author='Gagik',
    description='An alternative NN models using SoftH function',
    python_requires='>=3.7,<4',
    packages=find_packages(),
    install_requires=requirements,
)
