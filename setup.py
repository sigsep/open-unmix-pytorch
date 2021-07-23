from setuptools import setup, find_packages

umx_version = "1.2.1"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openunmix",
    version=umx_version,
    author="Fabian-Robert Stöter",
    author_email="fabian-robert.stoter@inria.fr",
    url="https://github.com/sigsep/open-unmix-pytorch",
    description="PyTorch-based music source separation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    install_requires=["numpy", "torchaudio>=0.9.0", "torch>=1.9.0", "tqdm"],
    extras_require={
        "asteroid": ["asteroid-filterbanks>=0.3.2"],
        "tests": [
            "pytest",
            "musdb>=0.4.0",
            "museval>=0.4.0",
            "asteroid-filterbanks>=0.3.2",
            "onnx",
            "tqdm",
        ],
        "stempeg": ["stempeg"],
        "evaluation": ["musdb>=0.4.0", "museval>=0.4.0"],
    },
    entry_points={"console_scripts": ["umx=openunmix.cli:separate"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
