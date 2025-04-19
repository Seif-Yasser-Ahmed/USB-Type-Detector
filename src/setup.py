from setuptools import setup, find_packages


def read_file(path):
    with open(path) as file:
        return file.read().strip().splitlines()


requirements = read_file('requirements.txt')

setup(
    name="USBTypeDetector",
    version="0.6",
    description="A Python package to detect USB type and provide information about the connected device.",
    author="EL SAKA",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "USBTypeDetector.pipeline.utils": ["*.yaml"],
    },
    install_requires=requirements,
    zip_safe=False,
)
