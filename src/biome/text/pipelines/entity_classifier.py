from typing import Union, List, Tuple

from allennlp.predictors import Predictor

from .pipeline import Pipeline
from biome.text.dataset_readers.entity_classifier_reader import EntityClassifierReader
from biome.text.models.sequence_classifier import SequenceClassifier


class EntityClassifierPipeline(Pipeline[SequenceClassifier, EntityClassifierReader]):
    # pylint: disable=arguments-differ
    def predict(
        self,
        kind: str,
        context: Union[str, List[List[str]]],
        position: Tuple[int, int],
        column_header: str = None,
    ):
        """
        This methods just define the api use for the model

        Parameters
        ----------
        kind
            A string defining the kind of the context: either "tabular" or "textual"
        context
            Either a text or a list of lists representing a table.
        position
            In the case of a textual context, it is the char_start and char_end position of the entity to be classified.
            In the case of a tabular context, it is the cell and row position.
        column_header
            In a tabular context, this is the column header of the entity to be classified.

        Returns
        -------
            The prediction result
        """
        return super(EntityClassifierPipeline, self).predict(
            kind=kind, context=context, position=position, column_header=column_header
        )


Predictor.register("entity_classifier")(EntityClassifierPipeline)
