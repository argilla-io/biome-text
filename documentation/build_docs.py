#!/usr/bin/env python3

import glob
import os
import subprocess
import sys

import click
import yaml

EXCLUDE_FILES = [
    ".DS_Store",
    "__init__.py",
    "__init__.pyc",
    "README.md",
    "version.py",
    "__main__.py",
]


def excluded_path(file_path: str) -> bool:
    return (
            "__pycache__" in file_path
            or "tests" in file_path
            or "mypy_cache" in file_path
            or os.path.basename(file_path) in EXCLUDE_FILES
    )


def pydocmd_configuration(base_dir: str, source_path: str, apidoc: str) -> None:
    pydocmd_config = {
        "generate": [],
        "additional_search": "../",
        "docs_dir": "sources",
        "gens_dir": apidoc,
        "headers": "markdown",
        "loader": "pydocmd.loader.PythonLoader",
        # preprocessor: kiposeq.utils.YamlPreprocessor
        "preprocessor": "pydocmd.preprocessors.simple.Preprocessor",
    }

    for file_path in glob.glob(f"{source_path}/**/*.py", recursive=True):
        if excluded_path(file_path):
            continue

        normalized_path = os.path.relpath(file_path, source_path)
        markdown_file = normalized_path.replace(".py", ".md")
        namespace = normalized_path.replace(".py", "").replace("/", ".")
        pydocmd_config["generate"].append({markdown_file: f"{namespace}++"})

    with open(os.path.join(base_dir, "pydocmd.yml"), "w") as pydocml_yaml:
        yaml.safe_dump(pydocmd_config, pydocml_yaml)


def mkdocs_navigations(base_dir: str, apidoc: str, mkdocs_apidoc_key: str = "API") -> None:
    apidoc_basepath = os.path.join(base_dir, apidoc)
    apidoc_navigation = []
    for markdown_file in glob.glob(f"{apidoc_basepath}/**/*.md", recursive=True):
        namespace = markdown_file.replace(f"{apidoc_basepath}/", "").replace(".md", "").replace("/", ".")
        reference_file = markdown_file.replace(apidoc_basepath, os.path.basename(apidoc))
        apidoc_navigation.append({namespace: reference_file})

    apidoc_navigation.sort(key=lambda x: list(x)[0], reverse=False)

    mkdocs_yaml = os.path.join(base_dir, "mkdocs.yml")
    with open(mkdocs_yaml) as mkdocs_config:
        mkdocs_config = yaml.safe_load(mkdocs_config)

    # Find the yaml corresponding to the API
    for nav_obj in mkdocs_config["nav"]:
        if mkdocs_apidoc_key in nav_obj:
            nav_obj[mkdocs_apidoc_key] = apidoc_navigation
            break

    with open(mkdocs_yaml, "w") as f:
        yaml.dump(mkdocs_config, f)


def apidoc_generation(base_dir: str):
    proc = subprocess.run(
        ["pydocmd", "generate"],
        stdout=subprocess.PIPE,
        universal_newlines=True,
        cwd=base_dir
    )


@click.command("build-docs")
@click.option("-s", "--source", "source_path", type=click.Path(exists=True), default="../src", show_default=True)
@click.option("-b", "--base-dir", "base_dir", type=click.Path(exists=True), default=".", show_default=True)
def build_docs(source_path: str, base_dir: str):
    click.echo("Building the docs...")
    sys.path.insert(0, source_path)
    apidoc = "docs/api"
    pydocmd_configuration(base_dir, source_path, apidoc)
    apidoc_generation(base_dir)
    mkdocs_navigations(base_dir, apidoc)


if __name__ == "__main__":
    build_docs()
