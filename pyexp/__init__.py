"""
This file is run when scripts are run under the "-m" module flag.
This is a sample package description. 
"""

### Sets up dependencies on other custom packages. [VCS_ROOT/Package]

# import sys
import os

VCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Package'))

### Checks requirements [TODO]
