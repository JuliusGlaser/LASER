from setuptools import setup, find_packages

setup(
    name="LASER",  # Change this to your package name
    version="1.0.0",
    author="Julius Glaser",
    author_email="julius-glaser@gmx.de",
    description="Implementation of the latent space decoded reconstruction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JuliusGlaser/LASER",
    packages=find_packages(),
    install_requires=[
        "numpy",  # Add required dependencies here
        "dipy",
        "torch",
        "PyYAML",
        "sigpy @ git+https://github.com/ZhengguoTan/sigpy.git@master"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)