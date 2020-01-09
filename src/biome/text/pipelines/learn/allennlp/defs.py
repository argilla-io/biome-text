import os
from typing import Dict, Any, cast

import yaml

from biome.text.environment import CUDA_DEVICE
from biome.text.pipelines.learn.default_callback_trainer import DefaultCallbackTrainer


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
    DATASET_READER_FIELD = "dataset_reader"
    TYPE_FIELD = "type"

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
        if self.model_path:
            self.model_dict = self.yaml_to_dict(self.model_path)
            self._model_to_new_format()  # for backward compatibility
        else:
            self.model_dict = {}
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

        model_dict = {}
        for _from, _to in [
            ("pipeline", self.DATASET_READER_FIELD),
            ("architecture", self.MODEL_FIELD),
        ]:
            model_dict[_to] = self.model_dict.get(_to, self.model_dict.get(_from))

        self.model_dict = model_dict or self.model_dict

        # In general the dataset reader should be of the same type as the model
        # (the DatasetReader's text_to_instance matches the model's forward method)
        if self.TYPE_FIELD not in self.model_dict.get(self.DATASET_READER_FIELD):
            self.model_dict[self.DATASET_READER_FIELD][
                self.TYPE_FIELD
            ] = self.model_dict[self.MODEL_FIELD][self.TYPE_FIELD]

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

    @staticmethod
    def get_cuda_device() -> int:
        """Gets the cuda device from an environment variable.

        This is necessary to activate a GPU if available

        Returns
        -------
        cuda_device
            The integer number of the CUDA device
        """
        cuda_device = int(os.getenv(CUDA_DEVICE, "-1"))
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

        trainer_cfg = cast(dict, allennlp_params[self.TRAINER_FIELD])
        if not trainer_cfg.get(self.TYPE_FIELD):  # Just in case of default trainer
            trainer_cfg[self.TYPE_FIELD] = DefaultCallbackTrainer.__name__
            allennlp_params[self.EVALUATE_ON_TEST_FIELD] = False
        # There is a bug in the AllenNLP train command: when specifying explicitly `type: default`, it will fail.
        if trainer_cfg[self.TYPE_FIELD] == "default":
            del trainer_cfg[self.TYPE_FIELD]

        return allennlp_params
