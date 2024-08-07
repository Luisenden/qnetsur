# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'qnetsur'
copyright = '2024, Luise Prielinger'
author = 'Luise Prielinger'
release = '0.0.1'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.extlinks',  # for external links
    'myst_parser', 'sphinx.ext.autosectionlabel'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
# MathJax configuration
mathjax3_config = {
    'TeX': {
        'extensions': ['AMSmath.js', 'AMSsymbols.js', 'color.js'],
        'Macros': {
            'Bar': ['\overline{#1}', 1],
            'dots': '\ldots',
            'mathbf': ['\mathbf{#1}', 1]
        }
    }
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes", ]
html_static_path = ['_static']
