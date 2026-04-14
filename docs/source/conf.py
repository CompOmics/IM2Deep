"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

from im2deep import __version__

# Project information
project = "im2deep"
author = "CompOmics"
github_project_url = "https://github.com/CompOmics/IM2Deep/"
github_doc_root = "https://github.com/CompOmics/IM2Deep/tree/main/docs/"

# Version
release = __version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_click.ext",
    "sphinx_rtd_theme",
    "sphinx_mdinclude",
]
source_suffix = [".rst", ".md"]
master_doc = "index"
exclude_patterns = ["_build"]

# Options for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Autodoc options
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "init"

# Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "psm_utils": ("https://psm-utils.readthedocs.io/en/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# Napoleon options
napoleon_numpy_docstring = True
napoleon_google_docstring = False
