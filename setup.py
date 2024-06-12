from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    include_package_data=True,
    name='qnetsur',  
    version='0.0.1',   
    description='Optimization of quantum network simulation parameters using surrogates.',
    author='Luise Prielinger',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements
)