import argparse
import logging
from typing import Dict

import coloredlogs
from allennlp.commands import Subcommand
from allennlp.common.util import import_submodules

from biome.text.allennlp.commands import BiomeExplore
from biome.text.allennlp.commands import BiomeLearn
from biome.text.allennlp.commands.predict import BiomePredict
from biome.text.allennlp.commands import BiomeRestAPI
from biome.text.allennlp.commands.vocab import BiomeVocab

command_name = "biome"


def configure_colored_logging(loglevel):
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["info"] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )


def main(
    prog: str = None, subcommand_overrides: Dict[str, Subcommand] = dict()
) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    parser = argparse.ArgumentParser(
        description="Run biome", usage="%(prog)s", prog=prog
    )

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands = {
        # Superseded by overrides
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != "configure":
            subparser.add_argument(
                "--include-package",
                type=str,
                action="append",
                default=[],
                help="additional packages to include",
            )

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, "include_package", ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    configure_colored_logging(loglevel=logging.INFO)
    main(
        command_name,
        subcommand_overrides=dict(
            predict=BiomePredict(),
            explore=BiomeExplore(),
            serve=BiomeRestAPI(),
            learn=BiomeLearn(),
            vocab=BiomeVocab(),
        ),
    )
