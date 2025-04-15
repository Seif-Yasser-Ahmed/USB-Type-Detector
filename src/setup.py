from setuptools import setup, find_packages

def read_file(path):
    with open(path) as file:
        return file.read().strip().splitlines()
    
requirements = read_file('requirements.txt')

setup(
    name="USBTypeDetector",
    version="0.5",
    description="A Python package to detect USB type and provide information about the connected device.",
    author="EL SAKA",
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)
