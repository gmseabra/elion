# Elion
An AI-based workflow for drug lead optimization

## Installation
```
### Conda Environment
Use the environemnt file provided, or the spec-file create for RHEL8.0

** Using the environemnt.yml file **
Follwing the instructinos from Conda docs, [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
```
$ conda env create -f environment.yml
```

** Using the spec file (RHEL 8)
Follow th einstructions from the Conda docs, [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments).
```
$ conda create --name myenv --file spec-file.txt
```


### Install Dependencies
$ cd to/project/folder/
$ conda activate elion
$ pip install --editable .
Obtaining file:///home/seabra/work/li/elion_dev/elion
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Installing backend dependencies ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: elion
  Building editable for elion (pyproject.toml) ... done
  Created wheel for elion: filename=elion-0.0.1a0.dev0-0.editable-py3-none-any.whl size=1740 sha256=668210deaf00e7bfc611743297e6e4b3e9e34a0198673054b137d824c0212041
  Stored in directory: /tmp/pip-ephem-wheel-cache-4o96z9y9/wheels/6b/da/ed/4f4d89acdea49ea9a18ebf1b66b36cbaaee17ea8f4c0eccc96
Successfully built elion
Installing collected packages: elion
Successfully installed elion-0.0.1a0.dev0
```

## Use
1. To get help information, type `elion --help`
1. Prepare an input file that controls the calculation to be done. By default, Elion looks for a file named `input.yml`. If you want to use any other name, just use the `-i` flag.
1. Run Elion with the command:
```
$ elion [-i input_file]
```

# Extra Info
This project uses `setuptools` and runs in editable mode, as described [here](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

# TO-DO
- input_reader.py (78): Restore the threshold values from the last iteration
- Property.py (185): Allow larger threshold jumps based on percentile
- Property.py (180): (bug) enforce threshold limit. Currenlty only works for increasing absolute thresholds.
