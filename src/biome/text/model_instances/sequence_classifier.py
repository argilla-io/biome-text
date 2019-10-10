from typing import Union, List

from biome.text import BaseModelInstance


class SequenceClassifier(BaseModelInstance):
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
