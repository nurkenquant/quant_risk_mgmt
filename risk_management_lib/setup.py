from setuptools import setup, find_packages

setup(
    name='risk_management_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas'
    ],
    description='A Python library for quantitative risk management',
    author='Nurken Abeuov'
)
