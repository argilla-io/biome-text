from typing import Dict, Optional, Any, List
import warnings
from biome.data.utils import get_nested_property_from_data
from numpy import number

GOLD_LABEL_DEFINITION_FIELD = "target"
VALUE_MAPPING_FIELD = "values_mapping"
USE_MISSING_LABEL_FIELD = "use_missing_label"
METADATA_FILE_FIELD = "metadata_file"

RESERVED_FIELD_PREFIX = "@"
SOURCE_FIELD = "{}source".format(RESERVED_FIELD_PREFIX)

# TODO: This file is not used at the moment!!! It should go away once we finished the forward refactoring and
#       made the predict command work!


class TransformationConfig(object):
    def __init__(
        self,
        field: Optional[str] = None,
        fields: List[str] = None,
        values_mapping: Dict[str, str] = None,
        use_missing_label: Optional[str] = None,
        metadata_file: Optional[str] = None,
    ):

        warnings.warn("This class won't be available anymore", DeprecationWarning)
        self.fields = [field] if field else fields
        self.value_mappings = (
            self.__mapping_from_metadata(metadata_file)
            if metadata_file
            else values_mapping
        )
        self.use_missing_label = use_missing_label

    @staticmethod
    def __mapping_from_metadata(path: str) -> Dict[str, str]:
        with open(path) as metadata_file:
            classes = [line.rstrip("\n").rstrip() for line in metadata_file]

        mapping = {idx + 1: cls for idx, cls in enumerate(classes)}
        # mapping variant with integer numbers
        mapping = {**mapping, **{str(key): value for key, value in mapping.items()}}

        return mapping


class ExamplePreparator(object):
    def __init__(self, dataset_transformations: Dict):
        warnings.warn("This class won't be available anymore", DeprecationWarning)
        transformations = (dataset_transformations or dict()).copy()
        gold_label_definition = transformations.pop(GOLD_LABEL_DEFINITION_FIELD, {})

        # for backwards compatibility
        if "gold_label" in gold_label_definition.keys():
            gold_label_definition["label"] = gold_label_definition.pop("gold_label")

        self._input_transformations = {
            key: self.__build_transformation(value)
            for key, value in transformations.items()
        }

        if gold_label_definition:
            values_mapping = gold_label_definition.pop(VALUE_MAPPING_FIELD, None)
            use_missing_label = gold_label_definition.pop(USE_MISSING_LABEL_FIELD, None)
            metadata_file = gold_label_definition.pop(METADATA_FILE_FIELD, None)

            for key, value in gold_label_definition.items():
                transformation = dict(
                    fields=[value],
                    values_mapping=values_mapping,
                    use_missing_label=use_missing_label,
                    metadata_file=metadata_file,
                )
                self._input_transformations[key] = self.__build_transformation(
                    transformation
                )

    @staticmethod
    def __build_transformation(transformation: Any) -> TransformationConfig:
        if isinstance(transformation, dict):
            return TransformationConfig(**transformation)
        if isinstance(transformation, list):
            return TransformationConfig(fields=transformation)
        return TransformationConfig(fields=[transformation])

    @staticmethod
    def __with_mapping(
        value: Any,
        mapping: Dict[str, str] = None,
        use_missing_label: Optional[str] = None,
    ):
        def sanitize(value: Any):
            if not value:
                return None
            if isinstance(value, float) or isinstance(value, number):
                return str(int(value))
            return str(value).strip()

        value = sanitize(value)
        value = None if not value or value.isspace() else value
        label = mapping.get(value, value) if mapping else value
        return (
            str(label).strip()
            if label
            else use_missing_label
            if use_missing_label
            else label
        )

    def __apply_transformation(
        self, example: Dict, transformation: TransformationConfig
    ) -> str:
        input_values = [
            self.__with_mapping(
                get_nested_property_from_data(example, input),
                mapping=transformation.value_mappings,
                use_missing_label=transformation.use_missing_label,
            )
            for input in transformation.fields
        ]

        input = " ".join([value for value in input_values if value]).strip()
        return input if len(input) > 0 else None

    def read_info(self, source: Dict, include_source: bool = False) -> Dict:

        example = {
            field_name: self.__apply_transformation(source, transformation_config)
            for field_name, transformation_config in self._input_transformations.items()
        }

        if not example:
            example = source
        elif include_source:
            example = {**example, SOURCE_FIELD: source}

        return example
