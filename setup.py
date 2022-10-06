import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = "0.0.1"

setuptools.setup(
    name="Heavy-ball_polyak_steps",
    version=version,
    author="Baptiste Goujaud, Adrien Taylor and Aymeric Dieuleveut",
    author_email="baptiste.goujaud@gmail.com",
    description="An adaptive Heavy-ball method based on Polyak step-size",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy", "matplotlib"],
    url="https://github.com/bgoujaud/Heavy-ball_polyak_steps",
    project_urls={
        "Documentation": "https://github.com/bgoujaud/Heavy-ball_polyak_steps/blob/master/README.md",
    },
    download_url="https://github.com/bgoujaud/Heavy-ball_polyak_steps/archive/refs/heads/master.zip",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[element for element in setuptools.find_packages() if element.startswith('Optimization')],
    python_requires=">=3.6",
)
