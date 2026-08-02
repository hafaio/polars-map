"""Sphinx conf."""

import sys
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from os import path

sys.path.append(path.abspath(".."))

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

project = "polars-map"
version = package_version(project)
release = version

copyright = f"{datetime.now(UTC).year:d} Erik Brinkman"  # noqa: A001
author = "Erik Brinkman"
