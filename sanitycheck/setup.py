from setuptools import setup, find_packages

setup(
    name="sanitycheck-cli",
    version="0.1.2",
    description="Zero-tuning CLI tool for quick numeric data sanity checks",
    long_description=open("./README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nexo-kun",
    author_email="nexokun.contact@gmail.com",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.21",
    ],
    entry_points={
        "console_scripts": [
            "sanitycheck = sanitycheck.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
