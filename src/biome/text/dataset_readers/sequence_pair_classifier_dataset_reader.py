import logging
from typing import Optional, Union, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField
from biome.text.dataset_readers.datasource_reader import DataSourceReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sequence_pair_classifier")
class SequencePairClassifierReader(DataSourceReader):
    """
    A DatasetReader for the SequencePairClassifier model.
    """

    def text_to_instance(
        self,
        record1: Union[str, List[str], dict],
        record2: Union[str, List[str], dict],
        label: Optional[str] = None,
    ) -> Optional[Instance]:

        fields = {}

        record1_field = self.build_textfield(record1)
        record2_field = self.build_textfield(record2)
        label_field = None
        # TODO: This is ugly as f***, should go into a decorator that checks/transforms the input
        if label is not None:
            # skipp example
            if str(label).strip() == "":
                return None
            else:
                label_field = LabelField(str(label).strip())

        if record1_field:
            fields["record1"] = record1_field

        if record2_field:
            fields["record2"] = record2_field

        if label_field:
            fields["label"] = label_field

        return Instance(fields) if fields else None


# Register an alias for this reader
DatasetReader.register("similarity_classifier")(SequencePairClassifierReader)
DatasetReader.register("biome_bimpm")(SequencePairClassifierReader)
