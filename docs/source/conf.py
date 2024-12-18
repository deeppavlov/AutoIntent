# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import version
from pathlib import Path

from docs.source.docs_utils.tutorials import generate_tutorial_links_for_notebook_creation

conf_dir = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120

sys.path.insert(0, conf_dir)

from docs_utils.skip_members import skip_member  # noqa: E402
from docs_utils.versions_generator import generate_versions_json  # noqa: E402

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
    "sphinx_multiversion",
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
BASE_STATIC_URL = f"{BASE_URL}/dev/_static"

html_theme_options = {
    "logo": {
        "text": "AutoIntent",
        "image_light": f"{BASE_STATIC_URL}/logo-light.svg",
        "image_dark": f"{BASE_STATIC_URL}/logo-dark.svg",
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
            "icon": f"{BASE_STATIC_URL}/hf-logo.svg",
            "type": "local",
        },
    ],
    "switcher": {
        "json_url": f"{BASE_STATIC_URL}/versions.json",
        "version_match": version,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "show_toc_level": 3,
}

html_favicon = f"{BASE_STATIC_URL}/logo-white.svg"
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

# sphinx_multiversion
# Whitelist for tags matching v1.0.0, v2.1.0 format
# smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'
smv_tag_whitelist = r"^.*$"

# Whitelist for the dev branch
smv_branch_whitelist = r"^dev$"

# Output format (keeping your current format)
smv_outputdir_format = "versions/{ref.name}"

# Include both tags and dev branch as released
smv_released_pattern = r"^(refs/tags/.*|refs/heads/dev)$"

smv_remote_whitelist = r"^(origin|upstream)$"  # Use branches from origin and upstream

repo_root = Path(__file__).resolve().parents[2]  # if conf.py is in docs/


def setup(app) -> None:  # noqa: ANN001
    generate_versions_json(repo_root, BASE_URL)

    generate_tutorial_links_for_notebook_creation(
        include=[
            (
                "user_guides.basic_usage",
                "Basic Usage",
            ),
            (
                "user_guides.advanced",
                "Advanced Usage",
            ),
            ("user_guides.cli", "CLI Usage"),
        ],
        source="user_guides",
        destination="docs/source/user_guides",
    )
    app.connect("autoapi-skip-member", skip_member)
