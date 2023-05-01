from setuptools import setup, find_packages

setup(
    name = "FibreAnalysis", 
    version = "0.1", 
    packages = find_packages(), 
    install_requires = ['scikit-image', 
                        'imagecodecs']
    )