# Elion
An AI-based workflow for drug lead optimization

## Installation
```
[Install dependencies]
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
