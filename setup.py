from setuptools import find_packages, setup


setup(
    name="perturb-subnet",
    version="0.1.0",
    description="Perturb Bittensor subnet (adversarial image attacks)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "bittensor>=8.0.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
    ],
)

