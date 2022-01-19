import setuptools


requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="web_app",
    version="0.0.1",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
)
