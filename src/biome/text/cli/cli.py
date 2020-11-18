import os
import sys

import click
from click import Group

from .serve import serve
from .train import train

SUPPORTED_COMMANDS = [serve, train]


def main():
    _add_project_modules_to_sys_path()

    commands = Group(no_args_is_help=True)
    for command in SUPPORTED_COMMANDS:
        commands.add_command(command, command.name)
    click.CommandCollection(sources=[commands])()


def _add_project_modules_to_sys_path():
    """This methods allows load udf defined from project location"""
    sys.path.append(os.getcwd())
