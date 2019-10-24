import os
import logging
from typing import Dict, Any
from biome.data.utils import get_nested_property_from_data

import yaml

from ..environment import CUDA_DEVICE

_logger = logging.getLogger(__name__)


class BiomeConfig:
    """This class contains biome config parameters usually necessary to run the biome commands.

    It also allows a transformation of these parameters to AllenNLP parameters.

    Parameters
    ----------
    model_path
        Path to the model yaml file
    trainer_path
        Path to the trainer yaml file
    vocab_path
        Path to the vocab yaml file
    train_path
        Path to the data source yaml file of the training set
    validation_path
        Path to the data source yaml file of the validation set
    test_path
        Path to the data source yaml file of the test set
    """

    # AllenNLP param fields
    CUDA_DEVICE_FIELD = "cuda_device"
    MODEL_FIELD = "model"
    TRAINER_FIELD = "trainer"
    TRAIN_DATA_FIELD = "train_data_path"
    VALIDATION_DATA_FIELD = "validation_data_path"
    TEST_DATA_FIELD = "test_data_path"
    EVALUATE_ON_TEST_FIELD = "evaluate_on_test"

    def __init__(
        self,
        model_path: str = None,
        trainer_path: str = None,
        vocab_path: str = None,
        train_path: str = None,
        validation_path: str = None,
        test_path: str = None,
    ):
        self.model_path = model_path
        self.trainer_path = trainer_path
        self.vocab_path = vocab_path
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path

        # Read yaml configs
        self.model_dict = self.yaml_to_dict(self.model_path)
        self._model_to_new_format()  # for backward compatibility
        self.vocab_dict = self.yaml_to_dict(self.vocab_path)
        self.trainer_dict = self.yaml_to_dict(self.trainer_path)
        # Add cuda device if necessary
        if not self.trainer_dict[self.TRAINER_FIELD].get(self.CUDA_DEVICE_FIELD):
            self.trainer_dict[self.TRAINER_FIELD][
                self.CUDA_DEVICE_FIELD
            ] = self.get_cuda_device()

    def _model_to_new_format(self):
        """This helper function transforms old model spec formats to the current one.

        The current one follows the AllenNLP format -> keys are "dataset_reader" and "model"

        Returns
        -------
        model_dict
            The model configuration as a dict
        """
        if "topology" in self.model_dict.keys():
            self.model_dict = dict(
                dataset_reader=get_nested_property_from_data(
                    self.model_dict, "topology.pipeline"
                ),
                model=get_nested_property_from_data(
                    self.model_dict, "topology.architecture"
                ),
            )
        # In general the dataset reader should be of the same type as the model
        # (the DatasetReader's text_to_instance matches the model's forward method)
        if "type" not in self.model_dict["dataset_reader"]:
            self.model_dict["dataset_reader"]["type"] = self.model_dict["model"]["type"]

    @staticmethod
    def yaml_to_dict(path: str) -> Dict[str, Any]:
        """Reads a yaml file and returns a dict.

        Parameters
        ----------
        path
            Path to the yaml file

        Returns
        -------
        dict
            If no path is specified, returns an empty dict
        """
        if not path:
            return dict()
        with open(path) as model_file:
            return yaml.safe_load(model_file)

    def get_cuda_device(self) -> int:
        """Gets the cuda device from an environment variable.

        This is necessary to activate a GPU if available

        Returns
        -------
        cuda_device
            The integer number of the CUDA device
        """
        cuda_device = int(os.getenv(CUDA_DEVICE, -1))
        return cuda_device

    def to_allennlp_params(self) -> Dict:
        """Transforms the cfg to AllenNLP parameters by basically joining all biome configurations.

        Returns
        -------
        params
            A dict in the right format containing the AllenNLP parameters
        """
        allennlp_params = dict(
            **self.model_dict, **self.trainer_dict, **self.vocab_dict
        )

        # add data fields
        data_fields = [
            self.TRAIN_DATA_FIELD,
            self.VALIDATION_DATA_FIELD,
            self.TEST_DATA_FIELD,
        ]
        data_paths = [self.train_path, self.validation_path, self.test_path]
        for field, path in zip(data_fields, data_paths):
            if path:
                allennlp_params[field] = path
        if self.test_path:
            allennlp_params[self.EVALUATE_ON_TEST_FIELD] = True

        return allennlp_params
