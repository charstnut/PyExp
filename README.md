# pyexp
This is a personal utility library and fitting framework built on `lmfit` and `scipy`. The goal is to put a bunch of common experimental analysis routines into a package, and possibly extend it with `Julia`. 
Common processing functions, such as calculating weighted averages, error propagation with `uncertainties`, etc., should be implemented in separate utility files (modules). 

Library structure [under package root `pyexp`]:
- fits
    - Fast fitting a model of choice (note: this is essentially a wrapper around scipy.optimize package) for non-histogram fits
    - A skeleton for histogram fits (binned and unbinned likelihood fits with proper error analysis and chi_2 test)
    - (If possible) Abstraction for MCMC utilities and framework
- `utils.py`
    - misc utility functions including drawing graphs, etc.
- `misc.py`
    - Other functions (e.g. self-implemented clustering)
- `avg.py`
    - Averaging related functions
- `demo_*.py`
    - Demonstration scripts


# Usage
Execute the module functions from python using the top-level 
pacakge:

```shell
python -m pyexp.avg
```

Or import the package:

```Python
import sys, os
# add other folders so libs can be imported
pyexp_dir = '/path/to/this/repo'
sys.path.append(pyexp_dir)

import pyexp.avg as avg
avg.function()
```

# Documentation
Please use standard docstring format for each function.

For vscode, use https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring with default doctring generation.
