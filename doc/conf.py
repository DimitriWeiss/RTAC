import os
import sys

# Add the module path (if needed)
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
project = 'rtac'
copyright = '2021, Dimitri Weiss'
author = 'Dimitri Weiss'
release = '0.0.1.dev'

# -- Sphinx Extensions ------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode', 
    'sphinxcontrib.bibtex',
]
master_doc = "index" 

autodoc_class_signature = "explicit"
autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # Include members without docstrings
    'show-inheritance': True,
    # "special-members": "__init__",  # Ensure constructors are documented
    # "exclude-members": "__weakref__",  # Avoid unnecessary members
    "inherited-members": False,
    # "autodoc_inherit_docstring": False,
    "exclude-members": "default",
    "special-members": "default",

}

exclude_patterns = [
    '_autosummary/rtac.tests.rst',
    '_autosummary/rtac.wrapper.cadical.rst',
    '_autosummary/rtac.main.rst',
    '_autosummary/rtac.feature_gen.cadical_feats.rst',
]

# -- HTML Output ------------------------------------------------------------
bibtex_bibfiles = ['references.bib']
html_theme = "sphinx_rtd_theme"
templates_path = ['_templates']
html_build_dir = '$READTHEDOCS_OUTPUT/html/'


def skip_member(app, what, name, obj, skip, options):
    skip_classes = {
        ("class", "Cadical", "rtac.wrapper.cadical"),
        ("class", "Cadicalpp", "rtac.wrapper.cadicalpp"),
    }
    print(f"SKIP CHECK: what={what}, name={name}, obj={obj}, module={getattr(obj, '__module__', None)}")
    if (what, name, getattr(obj, '__module__', '')) in skip_classes:
        return True
    if {getattr(obj, '__module__', None)} == 'module=rtac.wrapper.cadical':
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
