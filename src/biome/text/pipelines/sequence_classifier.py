from typing import Union, List, Type

import allennlp
from allennlp.predictors import Predictor

import biome
from biome.text.dataset_readers import SequenceClassifierReader
from biome.text.dataset_readers.datasource_reader import DataSourceReader
from .pipeline import Pipeline


class SequenceClassifier(Pipeline):
    @classmethod
    def reader_class(cls) -> Type[DataSourceReader]:
        return SequenceClassifierReader

    @classmethod
    def model_class(cls) -> Type[allennlp.models.Model]:
        return biome.text.models.SequenceClassifier

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
        return super(SequenceClassifier, self).predict(tokens=features)


Predictor.register("sequence_classifier")(SequenceClassifier)
