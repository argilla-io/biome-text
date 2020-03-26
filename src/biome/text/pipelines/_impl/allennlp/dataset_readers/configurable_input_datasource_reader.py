import copy
from typing import Dict, Union, List, Any, Optional, Iterable, cast, Tuple

from allennlp.data import Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import LabelField, MultiLabelField
from allennlp.data.tokenizers import SentenceSplitter

from biome.text.pipelines._impl.allennlp.dataset_readers import (
    DataSourceReader,
    TextTransforms,
)


class ConfigurableInputDatasourceReader(DataSourceReader):
    def __init__(
        self,
        inputs: List[str],
        output: str = "label",
        multi_label: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        segment_sentences: Union[bool, SentenceSplitter] = False,
        as_text_field: bool = True,
        skip_empty_tokens: bool = False,
        max_sequence_length: int = None,
        max_nr_of_sentences: int = None,
        text_transforms: TextTransforms = None,
    ):
        super(ConfigurableInputDatasourceReader, self).__init__(
            tokenizer,
            token_indexers,
            segment_sentences,
            as_text_field,
            skip_empty_tokens,
            max_sequence_length,
            max_nr_of_sentences,
            text_transforms,
        )

        self._inputs = inputs
        self._output = output

        self._multilabel = multi_label

    @property
    def inputs(self):
        return self._inputs.copy()

    @property
    def output(self):
        return self._output

    def text_to_instance(self, **inputs: Dict[str, Any]) -> Optional[Instance]:
        _inputs = copy.deepcopy(inputs)
        tokens = {input_key: _inputs.pop(input_key) for input_key in self._inputs}
        fields = {}

        tokens_field = self.build_textfield(tokens)
        if tokens_field:
            fields["tokens"] = tokens_field

        if self._output not in inputs:
            return Instance(fields) if fields else None

        label_data = inputs.get(self._output, "")
        if isinstance(label_data, (List, Tuple)):
            labels = [
                str(label).strip() for label in cast(Iterable, label_data) if label
            ]
        else:
            labels = [str(label_data).strip()]

        labels = [label for label in labels if label != ""]

        if len(labels) <= 0:
            return None

        fields["label"] = (
            MultiLabelField(labels)
            if self._multilabel
            else LabelField(str.join(" ", labels))
        )
        return Instance(fields)
