from setuptools import setup, find_packages

extras = {
    "dev": [
        "black",
        "pytest",
        "pylint",
        "twine",
        "jupyter",
        "jupyterlab",
        "matplotlib",
    ],
    "docs": ["sphinx", "sphinx-rtd-theme"],
}

setup(
    name="package_name",
    version="0.1.0",
    author="John Franklin Crenshaw",
    author_email="jfc20@uw.edu",
    description="Package description.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="http://github.com/jfcrenshaw/python-package",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy"],
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
