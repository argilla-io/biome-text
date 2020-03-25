from typing import Optional, Union, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField

from biome.text.pipelines._impl.allennlp.dataset_readers import DataSourceReader


@DatasetReader.register("sequence_classifier")
class SequenceClassifierReader(DataSourceReader):
    """
    A DatasetReader for the SequenceClassifier model.
    """

    # pylint: disable=arguments-differ
    def text_to_instance(
        self, tokens: Union[str, List[str], dict], label: Optional[str] = None
    ) -> Optional[Instance]:
        """Extracts the forward parameters from the example and transforms them to an `Instance`

        Parameters
        ----------
        tokens
            The input tokens key,values (or the text string)
        label
            The label value

        Returns
        -------
        instance
            Returns `None` if cannot generate an new Instance.
        """
        fields = {}

        tokens_field = self.build_textfield(tokens)
        if tokens_field:
            fields["tokens"] = tokens_field
        # TODO: Check how this affects predictions!
        elif self._skip_empty_tokens:
            return None

        label_field = None
        # TODO: This is ugly as f***, should go into a decorator that checks/transforms the input
        if label is not None:
            # skip example
            if str(label).strip() == "":
                return None
            label_field = LabelField(str(label).strip())

        if label_field:
            fields["label"] = label_field

        return Instance(fields) if fields else None


# Register an alias for this reader
DatasetReader.register("bert_for_classification")(SequenceClassifierReader)
