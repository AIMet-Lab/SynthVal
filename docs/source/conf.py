# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SynthVal'
copyright = '2024, Dario Guidotti'
author = 'Dario Guidotti'
release = '0.01a'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.napoleon',
]


autodoc_typehints = 'description'
templates_path = ['_templates']
exclude_patterns = []

autoapi_dirs = ['../../synthval']
autoapi_options = ['members',
                   'inherited-members:',
                   'private-members',
                   'show-inheritance',
                   'show-module-summary',
                   'special-members']

autoapi_own_page_level = 'class'
autoapi_ignore = ['*migrations*', '__init__.py']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']