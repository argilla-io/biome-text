import pytest
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

from biome.text.configuration import FeaturesConfiguration


def test_non_configurable_features():
    wrong_config = dict(ner=dict(embedding=15))
    with pytest.raises(TypeError):
        FeaturesConfiguration(**wrong_config)

    with pytest.raises(ConfigurationError):
        FeaturesConfiguration.from_params(Params(wrong_config))
