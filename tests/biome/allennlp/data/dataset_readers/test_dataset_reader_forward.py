import os
import pytest

import yaml
from allennlp.common import Params
from allennlp.data.fields import TextField, LabelField

from biome.allennlp.data.dataset_readers import ClassificationDatasetReader
from tests.test_context import TEST_RESOURCES, create_temp_configuration
from tests.test_support import DaskSupportTest

DS_READER_DEFINITION = os.path.join(TEST_RESOURCES,
                                    'resources/dataset_readers/definitions/forward_input_label_reader_definition.yml')

MULTIPLE_INPUT_READER_DEFINITION = os.path.join(TEST_RESOURCES,
                                                'resources/dataset_readers/definitions/forward_r1_r2_gold_reader_definition.yml')

JSON_PATH = os.path.join(TEST_RESOURCES, 'resources/data/dataset_source.jsonl')
NOLABEL_JSON_PATH = os.path.join(TEST_RESOURCES, 'resources/data/dataset_source_without_label.jsonl')

# TODO: Review the forward format in the DatasetReader and adapt tests!!!
class DatasetReaderForwardTest(DaskSupportTest):

    @pytest.mark.xfail
    def test_reader_with_forward_definition(self):
        with open(DS_READER_DEFINITION) as yml_config:
            reader = ClassificationDatasetReader.from_params(params=Params(yaml.load(yml_config)))
            read_config = dict(path=JSON_PATH, transformations=dict(input=['reviewText'], label='overall'))

            dataset = reader.read(create_temp_configuration(read_config))
            for example in dataset:
                input = example.fields.get('input')
                label = example.fields.get('label')

                assert input is not None, 'None input'
                assert label is not None, 'None label'

                assert isinstance(input, TextField), 'input is not a TextField instance'
                assert isinstance(label, LabelField), 'label is not a LabelField instance'

    @pytest.mark.xfail
    def test_reader_with_another_forward_definition(self):
        with open(MULTIPLE_INPUT_READER_DEFINITION) as yml_config:
            reader = ClassificationDatasetReader.from_params(params=Params(yaml.load(yml_config)))
            read_config = dict(path=JSON_PATH,
                               transformations=dict(r1=['reviewText'], r2=['reviewText'], gold='overall'))

            dataset = reader.read(create_temp_configuration(read_config))
            for example in dataset:
                r1 = example.fields.get('r1')
                r2 = example.fields.get('r2')
                gold = example.fields.get('gold')

                assert r1 is not None, 'None r1'
                assert r2 is not None, 'None r2'
                assert gold is not None, 'None gold'

                assert isinstance(r1, TextField), 'r1 is not a TextField instance'
                assert isinstance(r2, TextField), 'r2 is not a TextField instance'
                assert isinstance(gold, LabelField), 'gold is not a LabelField instance'

    @pytest.mark.xfail
    def test_reader_with_optional_params(self):
        with open(DS_READER_DEFINITION) as yml_config:
            reader = ClassificationDatasetReader.from_params(params=Params(yaml.load(yml_config)))
            read_config = dict(path=NOLABEL_JSON_PATH, transformations=dict(input=['reviewText'], label='overall'))

            dataset = reader.read(create_temp_configuration(read_config))
            for example in dataset:
                input = example.fields.get('input')
                label = example.fields.get('label')

                assert input is not None, 'None input'
                assert label is None, 'Expeced None label'

                assert isinstance(input, TextField), 'input is not a TextField instance'
