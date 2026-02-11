from setuptools import find_packages, setup

setup(
    name="synth-parallel",
    version="0.1.0",
    description="TranslateGemma-style synthetic parallel data pipeline",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=[
        "datasets>=2.19.0,<3.0.0",
        "openai>=1.40.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
        "typing-extensions>=4.10.0",
    ],
    extras_require={
        "test": ["pytest>=8.0.0"],
    },
    entry_points={
        "console_scripts": ["synth_parallel=synth_parallel.cli:main"],
    },
)
