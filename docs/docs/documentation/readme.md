# Installation

For the installation we recommend setting up a fresh [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```shell script
conda create -n biome python~=3.7.0 pip>=20.3.0
conda activate biome
```

Once the conda environment is activated, you can install the latest release or the development version via pip.

## Latest release (recommended)

To install the latest release of *biome.text* type in:

````shell script
pip install -U biome-text
````

After installing *biome.text*, the best way to test your installation is by running the *biome.text* cli command:

```shell script
biome --help
```

## Master branch

The *master branch* contains the latest features, but is less well tested.
If you are looking for a specific feature that has not been released yet, you can install the package from our master branch with:

````shell script
pip install -U git+https://github.com/recognai/biome-text.git
````
