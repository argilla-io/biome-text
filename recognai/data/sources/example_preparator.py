from typing import Dict, Optional, Iterable

GOLD_LABEL_DEFINITION_FIELD = 'target'
VALUE_MAPPING_FIELD = "values_mapping"
USE_MISSING_LABEL_FIELD = "use_missing_label"

DEFAULT_MISSING_LABEL = 'None'


class ExamplePreparator(object):
    def __init__(self, dataset_transformations: Dict):
        transformations = (dataset_transformations or {}).copy()
        gold_label_definition = transformations.pop(GOLD_LABEL_DEFINITION_FIELD, {})

        self._value_mappings = gold_label_definition.pop(VALUE_MAPPING_FIELD, None)
        self._use_missing_label = gold_label_definition.pop(USE_MISSING_LABEL_FIELD, None)
        self._gold_label_id = self._gold_label_definition(gold_label_definition)
        self._gold_label_field = gold_label_definition.get(self._gold_label_id, None)
        self._input_transformations = transformations

    def _gold_label_definition(self, gold_label_definition: Optional[Dict]):
        if gold_label_definition is None:
            return None
        else:
            for name in gold_label_definition.keys():
                return name

    def _input(self, example: Dict, fields: Iterable[str]) -> str:
        return " ".join([str(example[input]) for input in fields]).strip()

    def _gold_label(self, example: Dict) -> str:

        def with_mapping(value, mapping=None, use_missing_label: str = None):
            # Adding default value to value, enables partial mapping
            # Handling missing labels with a default value
            value = None if not value or value == '' else value
            label = mapping.get(value, value) if mapping else value
            return str(label).strip() if label \
                else use_missing_label if use_missing_label \
                else label

        label = with_mapping(
                example.get(self._gold_label_field),
                self._value_mappings,
                self._use_missing_label
            )

        return label

    def read_info(self, example: Dict) -> Dict:

        if self._input_transformations:

            mapped_example = {}
            for field_name, example_fields in self._input_transformations.items():
                mapped_example[field_name] = self._input(example, example_fields)
            if self._gold_label_id:
                mapped_example[self._gold_label_id] = self._gold_label(example)
            return mapped_example

        return example
