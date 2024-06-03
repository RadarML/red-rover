# Global configuration options

copyright = '2024, Wiselab'
author = 'Tianshu Huang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinxarg.ext',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Remove namedtuple "Alias for ...", typeddict duplicates
import collections, typing
def remove_namedtuple_attrib_docstring(app, what, name, obj, skip, options):
    if type(obj) is collections._tuplegetter:
        return True
    return skip


def setup(app):
    app.connect('autodoc-skip-member', remove_namedtuple_attrib_docstring)
