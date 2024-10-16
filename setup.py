import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SynthVal",
    version="0.1.3-alpha",
    author="Dario Guidotti, Laura Pandolfo, Luca Pulina",
    author_email="dguidotti@uniss.it, lpandolfo@uniss.it, lpulina@uniss.it",
    license='GNU General Public License with Commons Clause License Condition v1.0',
    description="Package designed to validate the quality of synthetically generated data, with a focus on medical "
                "images like chest x-rays and mammographies, by providing tools for feature extraction and similarity "
                "metric calculations to compare original and synthetic datasets.",
    long_description=long_description,
    url="https://github.com/AIMet-Lab/SynthVal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=['pillow', 'pydicom', 'numpy', 'pandas', 'timm', 'torch',
                      'transformers', 'scipy', 'dcor', 'pynever', 'scikit-learn'],
)