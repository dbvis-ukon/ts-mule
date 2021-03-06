# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from tsmule import __version__ as info

# if run `make html` from docs.
sys.path.insert(0, os.path.abspath('../../tsmule'))


# -- Project information -----------------------------------------------------
project = info.__title__
author = info.__author__
copyright = info.__copyright__
version = info.__version__
license = info.__license__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',      # parser for docstrings numpy/google
    # 'sphinx.ext.mathjax',     # requires mathjax/latex
    'IPython.sphinxext.ipython_console_highlighting',   # lexer ipython3
    'nbsphinx',                 # parser for *.ipynb files
]

source_suffix = ['.rst', '.ipynb']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. bizstyle
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for Napoleon extension -------------------------------------------
# napoleon_google_docstring = False

# To build module docs
# sphinx-apidoc -f -o <path-to-output> <path-to-module>

# -- Options for nbsphinx -------------------------------------------------
nbsphinx_allow_errors = True
