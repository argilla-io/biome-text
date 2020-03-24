from typing import Union, List, Optional, Tuple

from allennlp.data import Instance
from biome.text.dataset_readers.datasource_reader import DataSourceReader


class EntityClassifierReader(DataSourceReader):
    def text_to_instance(
        self,
        kind: str,
        context: Union[str, List[List[str]]],
        position: Tuple[int, int],
        entities: List[Tuple[int, int, str]] = None,
        column_header: str = None,
        label: str = None,
    ) -> Optional[Instance]:
        """

        Parameters
        ----------
        kind
            A string defining the kind of the context: either "tabular" or "textual"
        context
            Either a text or a list of lists representing a table.
        position
            In the case of a textual context, it is the char_start and char_end position of the entity to be classified.
            In the case of a tabular context, it is the cell and row position.
        entities
            A list of labels provided for the rest of the context tokens. Each entity is represented as a tuple:
            (char_start/cell position in the context, char_end/row position in the context, label)
            TODO: to be implemented
        column_header
            In a tabular context, this is the column header of the entity to be classified.
        label
            The label value

        Returns
        -------
        instance
        """
        if kind == "textual":
            # TODO: ask paco, is the end position inclusive or exclusive?
            entity = context[position[0] : position[1]]
        elif kind == "tabular":
            entity = context[0][position[0]]
        else:
            raise ValueError(
                f"'{kind}' is not a valid value for the 'kind' argument. "
                f"It has to be one of the following choices: ['textual', 'tabular']"
            )

        # For now we just pass on the input values as dict.
        # Once we have proper EntityClassifier model this will likely change
        token_dict = {
            "kind": kind,
            "context": self._value_as_string(context),
            "entity": entity,
        }
        # For now we force as_text_field to False
        if self._as_text_field:
            self._logger.warning(
                "For the EntityClassifier we set 'as_text_field' automatically to False."
            )
            self._as_text_field = False

        # add column header if provided
        if column_header:
            token_dict.update({"column_header": column_header})

        fields = {}

        tokens_field = self.build_textfield(token_dict)
        if tokens_field:
            fields["tokens"] = tokens_field
        if label is not None:
            fields["label"] = label

        return Instance(fields) if fields else None
