from setuptools import setup, find_packages

# setup.py가 패키지 디렉토리 안에 있으므로 (distributed_rl/setup.py),
# find_packages()는 하위 패키지(rewards 등)만 찾고 distributed_rl 자체는 빠진다.
# package_dir로 "distributed_rl" → "." 매핑을 명시해야 python -m distributed_rl이 동작한다.
_sub_packages = find_packages(exclude=["tests", "tests.*"])

setup(
    name="distributed_rl",
    version="0.1.0",
    packages=["distributed_rl"] + ["distributed_rl." + p for p in _sub_packages],
    package_dir={"distributed_rl": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0,<3.0.0",
        "transformers>=5.2.0",
        "trl>=0.15.0",
        "accelerate>=0.34.0",
        "deepspeed>=0.15.4",
        "datasets>=2.21.0",
        "sentencepiece>=0.2.0",
        "PyYAML>=6.0.1",
    ],
)
