from setuptools import find_packages, setup


setup(
    name="data-analysis-tool",
    version="0.1.0",
    description="Data analysis and plotting utilities",
    author="Sasindu Wijethunga",
    author_email="sasindusulochana99@gmail.com",
    long_description=open("README.md", encoding="utf-8").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Sasindu99-ai/data-analysis",
    license="MIT",
    packages=find_packages(include=["data_analysis", "data_analysis.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "ipython",
        "pygments",
        "scipy",
        "scikit-learn",
        "plotly",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="data-analysis data-inspection plotting",
    zip_safe=False,
)