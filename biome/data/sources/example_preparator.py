from typing import Dict, Optional, Any, List

GOLD_LABEL_DEFINITION_FIELD = 'target'
VALUE_MAPPING_FIELD = "values_mapping"
USE_MISSING_LABEL_FIELD = "use_missing_label"

RESERVED_FIELD_PREFIX = '@'
SOURCE_FIELD = 'source'


class TransformationConfig(object):

    def __init__(self,
                 field: Optional[str] = None,
                 fields: List[str] = None,
                 values_mapping: Dict[str, str] = None,
                 use_missing_label: Optional[str] = None):
        self.fields = [field] if field else fields
        self.value_mappings = values_mapping
        self.use_missing_label = use_missing_label


class ExamplePreparator(object):

    def __init__(self, dataset_transformations: Dict, include_source: bool = False):
        transformations = (dataset_transformations or dict()).copy()
        gold_label_definition = transformations.pop(GOLD_LABEL_DEFINITION_FIELD, {})

        self._include_source = include_source
        self._input_transformations = {
            key: self.__build_transformation(value) for key, value in transformations.items()
        }

        if gold_label_definition:
            values_mapping = gold_label_definition.pop(VALUE_MAPPING_FIELD, None)
            use_missing_label = gold_label_definition.pop(USE_MISSING_LABEL_FIELD, None)

            for key, value in gold_label_definition.items():
                transformation = dict(
                    fields=[value],
                    values_mapping=values_mapping,
                    use_missing_label=use_missing_label)
                self._input_transformations[key] = self.__build_transformation(transformation)

    @staticmethod
    def __build_transformation(transformation: Any) -> TransformationConfig:
        if isinstance(transformation, dict):
            return TransformationConfig(**transformation)
        if isinstance(transformation, list):
            return TransformationConfig(fields=transformation)
        return TransformationConfig(fields=[transformation])

    @staticmethod
    def __with_mapping(value: Any, mapping: Dict[str, str] = None, use_missing_label: Optional[str] = None):
        # Adding default value to value, enables partial mapping
        # Handling missing labels with a default value
        value = None if not value or str(value).isspace() else value
        label = mapping.get(value, value) if mapping else value
        return str(label).strip() if label \
            else use_missing_label if use_missing_label \
            else label

    def __apply_transformation(self, example: Dict, transformation: TransformationConfig) -> str:
        input_values = [self.__with_mapping(example.get(input),
                                            mapping=transformation.value_mappings,
                                            use_missing_label=transformation.use_missing_label)
                        for input in transformation.fields]

        input = " ".join([value for value in input_values if value]).strip()

        return input if len(input) > 0 else None

    def read_info(self, source: Dict) -> Dict:
        example = {
            field_name: self.__apply_transformation(source, transformation_config)
            for field_name, transformation_config in self._input_transformations.items()
        }

        if not example:
            example = source

        return example if not self._include_source else {
            **example,
            f'{RESERVED_FIELD_PREFIX}{SOURCE_FIELD}': source
        }
