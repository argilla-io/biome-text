import argparse
import logging

import coloredlogs
from allennlp.common.util import import_submodules

from .allennlp.commands import BiomeExplore
from .allennlp.commands import BiomeLearn
from .allennlp.commands import BiomePredict
from .allennlp.commands import BiomeRestAPI
from .allennlp.commands import BiomeVocab


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


def main() -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Run biome", prog="biome")

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands = {
        "predict": BiomePredict(),
        "explore": BiomeExplore(),
        "serve": BiomeRestAPI(),
        "learn": BiomeLearn(),
        "vocab": BiomeVocab(),
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
            logger.info("Loading packager {}".format(package_name))
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    configure_colored_logging(loglevel=logging.INFO)
    main()
