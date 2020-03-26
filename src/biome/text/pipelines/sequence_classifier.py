from typing import Union, List

from allennlp.predictors import Predictor

from .pipeline import Pipeline
from biome.text.pipelines._impl.allennlp.dataset_readers import SequenceClassifierReader
from biome.text.pipelines._impl.allennlp.models import SequenceClassifier


class SequenceClassifierPipeline(
    Pipeline[SequenceClassifier, SequenceClassifierReader]
):

    # pylint: disable=arguments-differ
    def predict(self, tokens: Union[dict, str, List[str]]):
        """
        This methods just define the api use for the model
        Parameters
        ----------
        tokens
            The data features used for prediction

        Returns
        -------
            The prediction result

        """
        return super(SequenceClassifierPipeline, self).predict(tokens=tokens)


Predictor.register("sequence_classifier")(SequenceClassifierPipeline)
