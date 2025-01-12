from setuptools import setup, find_packages

setup(
    name='llm_lab',  # Choose a name for your package
    version='0.1.0',  # Your package version
    packages=find_packages(where='src'),  # Find packages in the 'src' directory
    package_dir={'': 'src'},  # Tell setuptools that packages are under 'src'
    install_requires=[
        # List any dependencies your package needs
        'torch',
        'transformers',
        'datasets' 
    ],
)