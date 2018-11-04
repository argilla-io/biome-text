from allennlp.commands import main

from biome.commands.predict.predict import BiomePredict
from biome.commands.publish import BiomePublishModel
from biome.commands.restapi import BiomeRestAPI
from biome.commands.train import BiomeTrain

command_name = 'biome'

if __name__ == '__main__':
    main(command_name, subcommand_overrides=dict(learn=BiomeTrain(),
                                                 predict=BiomePredict(),
                                                 publish=BiomePublishModel(),
                                                 serve=BiomeRestAPI()))