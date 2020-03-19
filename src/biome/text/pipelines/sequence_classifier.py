from typing import Union, List

from allennlp.predictors import Predictor

from biome.text.dataset_readers.datasource_reader import DataSourceReader
from .pipeline import Pipeline
from ..models import SequenceClassifier


class SequenceClassifierPipeline(Pipeline[SequenceClassifier, DataSourceReader]):

    # pylint: disable=arguments-differ
    def predict(self, features: Union[dict, str, List[str]]):
        """
        This methods just define the api use for the model
        Parameters
        ----------
        features
            The data features used for prediction

        Returns
        -------
            The prediction result

        """
        return super(SequenceClassifierPipeline, self).predict(tokens=features)


Predictor.register("sequence_classifier")(SequenceClassifierPipeline)
