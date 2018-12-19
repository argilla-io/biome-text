import os
import unittest

from biome.data.sources.example_preparator import ExamplePreparator
from tests.test_context import TEST_RESOURCES

METADATA_FILE = os.path.join(TEST_RESOURCES, 'resources/classes.txt')


class ExamplePreparatorTest(unittest.TestCase):

    def test_simple_example_preparator(self):
        preparator = ExamplePreparator(dict(), include_source=False)

        source = dict(a='single', dictionary='example')
        example = preparator.read_info(source)

        assert source == example

    def test_simple_transformations(self):
        preparator = ExamplePreparator(dict(gold_label='a', other_field=['b', 'c']), include_source=False)

        source = dict(a='Target field', b='b value', c='c value')
        example = preparator.read_info(source)
        assert source['a'] == example['gold_label']
        assert example['other_field'] == ' '.join([source['b'], source['c']])

        source = dict(a='Target field', b='b value')
        example = preparator.read_info(source)
        assert source['a'] == example['gold_label']
        assert example['other_field'] == source['b']

        source = dict(a='Target field', c='c value')
        example = preparator.read_info(source)
        assert source['a'] == example['gold_label']
        assert example['other_field'] == source['c']

        source = dict(c='c value')
        example = preparator.read_info(source)
        assert not example['gold_label']

    def test_transformations_using_missing_values(self):
        preparator = ExamplePreparator(
            dict(gold_label=dict(field='a', use_missing_label='Missing'), other_field=['b', 'c']),
            include_source=True)

        source = dict(b='b value', c='c value')
        example = preparator.read_info(source)
        assert 'Missing' == example['gold_label']
        assert example['other_field'] == ' '.join([source['b'], source['c']])
        assert example['@source'] == source

        source = dict(a='A value', b='b value', c='c value')
        example = preparator.read_info(source)
        assert source['a'] == example['gold_label']
        assert example['other_field'] == ' '.join([source['b'], source['c']])

    def test_target_configuration(self):
        preparator = ExamplePreparator(dict(target=dict(gold_label='a'), other_field=['b', 'c']))

        source = dict(a='A value', b='b value', c='c value')
        example = preparator.read_info(source)
        assert source['a'] == example['gold_label']
        assert example['other_field'] == ' '.join([source['b'], source['c']])

    def test_mappings_from_metadata(self):
        preparator = ExamplePreparator(
            dict(gold_label=dict(field='a', metadata_file=METADATA_FILE), other_field=['b', 'c']))

        example = preparator.read_info(dict(a=1))
        assert example['gold_label'] == 'Class One'

        example = preparator.read_info(dict(a=2))
        assert example['gold_label'] == 'Class two'

        example = preparator.read_info(dict(a=3))
        assert example['gold_label'] == 'Another class'

        example = preparator.read_info(dict(a=4))
        assert example['gold_label'] == 'You b*****d'

        example = preparator.read_info(dict(a='4'))
        assert example['gold_label'] == 'You b*****d'
