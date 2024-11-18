# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


from autointent.docs_utils import (
    generate_tutorial_links_for_notebook_creation,
    regenerate_apiref,
)

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
]

templates_path = ["_templates"]
exclude_patterns = ["conf.py"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": True,
    # "special-members": "__call__",
    "member-order": "bysource",
    # "exclude-members": "_abc_impl, model_fields, model_computed_fields, model_config",
}

# Finding tutorials directories
nbsphinx_custom_formats = {".py": "autointent.docs_utils.py_percent_to_notebook"}
nbsphinx_prolog = """
:tutorial_name: {{ env.docname }}
"""


def setup(_) -> None:  # noqa: ANN001
    generate_tutorial_links_for_notebook_creation(
        [
            ("tutorials.pipeline_optimization", "Pipeline Optimization"),
        ]
    )
    regenerate_apiref(
        [
            ("autointent.configs", "Configs"),
            ("autointent.context", "Context"),
            ("autointent.generation", "Generation"),
            ("autointent.metrics", "Metrics"),
            ("autointent.modules", "Modules"),
            ("autointent.nodes", "Nodes"),
            ("autointent.pipeline", "Pipeline"),
        ]
    )
