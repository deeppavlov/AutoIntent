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
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.todo",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["conf.py", "docs_utils/*"]

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
nbsphinx_execute = "never"

todo_include_todos = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    # "special-members": "__call__",
    "member-order": "bysource",
    # "exclude-members": "_abc_impl, model_fields, model_computed_fields, model_config",
}

# Finding tutorials directories
nbsphinx_custom_formats = {".py": "docs_utils.notebook.py_percent_to_notebook"}
nbsphinx_prolog = """
:tutorial_name: {{ env.docname }}
"""


def setup(app) -> None:  # noqa: ANN001
    generate_tutorial_links_for_notebook_creation(
        [
            ("tutorials.pipeline_optimization", "Pipeline Optimization"),
            ("tutorials.modules.scoring", "Scoring Modules", [("linear", "Linear Scorer")]),
            ("tutorials.modules.prediction", "Prediction Modules", [("argmax", "Argmax Predictor")]),
        ]
    )
    app.connect("autodoc-skip-member", skip_member)
