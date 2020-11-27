import os
import sys

import click
from click import Group

from .evaluate import evaluate
from .serve import serve
from .train import train

SUPPORTED_COMMANDS = [train, evaluate, serve]


def main():
    _add_project_modules_to_sys_path()

    commands = Group(no_args_is_help=True)
    for command in SUPPORTED_COMMANDS:
        commands.add_command(command, command.name)
    click.CommandCollection(sources=[commands])()


def _add_project_modules_to_sys_path():
    """This methods allows load udf defined from project location"""
    sys.path.append(os.getcwd())
