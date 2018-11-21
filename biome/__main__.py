import logging

from allennlp.commands import main

from biome.allennlp.commands import BiomeRestAPI
from biome.allennlp.commands.explore.explore import BiomeExplore
from biome.allennlp.commands.learn import BiomeLearn

command_name = 'biome'


def configure_colored_logging(loglevel):
    import coloredlogs
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles['asctime'] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles['info'] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt='%(asctime)s %(levelname)-8s %(name)s  - %(message)s',
        level_styles=level_styles,
        field_styles=field_styles)


if __name__ == '__main__':
    configure_colored_logging(loglevel=logging.INFO)
    main(command_name,
         subcommand_overrides=dict(
             explore=BiomeExplore()
             , serve=BiomeRestAPI()
             , learn=BiomeLearn()
         ))
