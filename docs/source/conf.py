# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from autointent.docs_utils.regenerate_apiref import regenerate_apiref

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
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


def setup(_) -> None:  # noqa: ANN001
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
