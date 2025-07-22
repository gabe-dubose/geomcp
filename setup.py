from importlib.metadata import entry_points
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geomcp",
    version="2025.07",
    author="Gabe DuBose",
    author_email="james.g.dubose@gmail.com",
    description="GEOmetric Model of Conditional Phenotypes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabe-dubose/geomcp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    install_requires = ['matplotlib', 'numpy']
)