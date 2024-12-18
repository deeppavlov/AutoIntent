# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import json
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path
from importlib.metadata import version

conf_dir = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120

sys.path.insert(0, conf_dir)

from docs_utils.skip_members import skip_member  # noqa: E402
from docs_utils.tutorials import generate_tutorial_links_for_notebook_creation  # noqa: E402
from docs_utils.versions_generator import generate_versions_json, get_sorted_versions  # noqa: E402

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

# nitpicky = True  # warn about unknown links

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "datasets": ("https://huggingface.co/docs/datasets/master/en/", None),
    "transformers": ("https://huggingface.co/docs/transformers/master/en/", None),
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
version = version("autointent").replace("dev", "")  # may differ

BASE_URL = "https://deeppavlov.github.io/AutoIntent/versions"
BASE_STATIC_URL = "https://deeppavlov.github.io/AutoIntent/versions/dev/_static"

html_theme_options = {
    "logo": {
        "text": "AutoIntent",
        "image_light": f"{BASE_URL}/logo-light.svg",
        "image_dark": f"{BASE_URL}/logo-dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/deeppavlov/AutoIntent",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "HuggingFace",
            "url": "https://huggingface.co/AutoIntent",
            "icon": f"{BASE_URL}/hf-logo.svg",
            "type": "local",
        },
    ],
    "switcher": {
        "json_url": f"{BASE_URL}/versions.json",
        "version_match": version,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "show_toc_level": 3,
}

html_favicon = f"{BASE_URL}/logo-white.svg"
html_show_sourcelink = False

toc_object_entries_show_parents = "hide"

todo_include_todos = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    # "special-members": "__call__",
    "member-order": "groupwise",
    "exclude-members": "_abc_impl, model_fields, model_computed_fields, model_config",
}

# Finding tutorials directories
nbsphinx_custom_formats = {".py": "docs_utils.notebook.py_percent_to_notebook"}
nbsphinx_prolog = """
:tutorial_name: {{ env.docname }}
"""
# nbsphinx_execute = "never"
nbsphinx_thumbnails = {
    "user_guides/*": "_static/square-white.svg",
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))  # if conf.py is in docs/
def setup(app) -> None:  # noqa: ANN001
    generate_versions_json(app, repo_root, BASE_URL)
    # versions = get_sorted_versions()
    # with open('docs/source/_static/versions.json', 'w') as f:
    #     json.dump(versions, f, indent=4)
#   generate_tutorial_links_for_notebook_creation(
#         include=[
#             (
#                 "user_guides.basic_usage",
#                 "Basic Usage",
#             ),
#             (
#                 "user_guides.advanced",
#                 "Advanced Usage",
#             ),
#             ("user_guides.cli", "CLI Usage"),
#         ],
#         source="user_guides",
#         destination="docs/source/user_guides",
#     )
#     app.connect("autoapi-skip-member", skip_member)
