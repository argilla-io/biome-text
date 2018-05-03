from typing import Dict, Tuple

INPUTS_FIELD = 'inputs'
GOLD_LABEL = 'gold_label'

MISSING_LABEL_DEFAULT = 'None'


class ClassificationInstancePreparator(object):

    def __init__(self, dataset_transformations: Dict):
        self._dataset_transformations = dataset_transformations

    def _input(self, example: Dict) -> str:
        inputs_field = self._dataset_transformations[INPUTS_FIELD]
        return str(inputs_field \
                       if type(inputs_field) is str \
                       else " ".join([str(example[input]) for input in inputs_field])).strip()

    def _gold_label(self, example: Dict) -> str:
        def with_mapping(value, mapping=None, use_missing_label: str = None):
            # Adding default value to value, enables partial mapping
            # Handling missing labels with a default value
            value = None if not value or value == '' else value
            label = mapping.get(value, value) if mapping else value
            return str(label).strip() if label \
                else use_missing_label if use_missing_label \
                else label

        field_type = "field"
        field_mapping = "values_mapping"
        use_missing_label = "use_missing_label"

        gold_label_definition = self._dataset_transformations[GOLD_LABEL]

        label = (str(example.get(gold_label_definition))
                 if type(gold_label_definition) is str
                 else with_mapping(str(example.get(gold_label_definition[field_type], '')).strip(),
                                   gold_label_definition.get(field_mapping, None),
                                   gold_label_definition.get(use_missing_label, None)))

        return label

    def read_info(self, example: Dict) -> Tuple[str, str]:
        input_text = self._input(example)
        label = self._gold_label(example)
        return input_text, label
