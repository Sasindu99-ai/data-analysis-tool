from setuptools import find_packages, setup


setup(
    name="data-analysis-tool",
    version="0.1.0",
    description="Data analysis and plotting utilities",
    author="Sasindu Wijethunga",
    author_email="sasindusulochana99@gmail.com",
    packages=find_packages(include=["data_analysis", "data_analysis.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "ipython",
        "pygments",
        "scipy",
        "scikit-learn",
        "plotly",
    ],
)