# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

conf_dir = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120

sys.path.insert(0, conf_dir)

from docs_utils.skip_members import skip_member  # noqa: E402
from docs_utils.tutorials import generate_tutorial_links_for_notebook_creation  # noqa: E402

project = "AutoIntent"
copyright = "2024, DeepPavlov"
author = "DeepPavlov"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "autoapi.extension",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["conf.py", "docs_utils/*"]

# API reference

nitpicky = True  # warn about unknown links

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pathlib": ("https://docs.python.org/3/library/pathlib.html", None),
    # Add other mappings as needed
}

autoapi_keep_files = True
autoapi_dirs = [Path.cwd().parent.parent / "autointent"]
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autodoc_typehints = "description"
autoapi_own_page_level = "function"
suppress_warnings = ["autoapi.python_import_resolution"]
autoapi_add_toctree_entry = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["../_static"]

html_theme_options = {
    "logo": {
        "text": "AutoIntent",
        "image_light": "../_static/logo-light.svg",
        "image_dark": "../_static/logo-dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/deeppavlov/AutoIntent",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_toc_level": 3,
}

html_favicon = "../_static/logo-white.svg"
html_show_sourcelink = False

toc_object_entries_show_parents = "hide"

todo_include_todos = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    # "special-members": "__call__",
    "member-order": "bysource",
    "exclude-members": "_abc_impl, model_fields, model_computed_fields, model_config",
}

# Finding tutorials directories
nbsphinx_custom_formats = {".py": "docs_utils.notebook.py_percent_to_notebook"}
nbsphinx_prolog = """
:tutorial_name: {{ env.docname }}
"""
nbsphinx_execute = "always"

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"


def setup(app) -> None:  # noqa: ANN001
    generate_tutorial_links_for_notebook_creation(
        [
            ("tutorials.pipeline_optimization", "Pipeline Optimization"),
            ("tutorials.modules.scoring", "Scoring Modules", [("linear", "Linear Scorer")]),
            ("tutorials.modules.prediction", "Prediction Modules", [("argmax", "Argmax Predictor")]),
            ("tutorials.data", "Data"),
        ],
    )
    app.connect("autodoc-skip-member", skip_member)
