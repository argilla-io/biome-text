import copy
import logging
from dataclasses import asdict
from typing import Optional
from typing import Union

import pytorch_lightning as pl
from allennlp.common import Params
from allennlp.data import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training.optimizers import Optimizer
from torch.utils.data import IterableDataset

from biome.text.configuration import LTrainerConfiguration
from biome.text.configuration import VocabularyConfiguration
from biome.text.dataset import Dataset
from biome.text.dataset import InstancesDataset
from biome.text.pipeline import Pipeline
from biome.text.training_results import TrainingResults

_LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: LTrainerConfiguration,
    ):
        self._config = copy.deepcopy(config)

        # remove non pl.Trainer arguments
        trainer_kwargs = asdict(self._config)
        for kwarg in ["num_epochs", "optimizer", "data_bucketing", "batch_size"]:
            del trainer_kwargs[kwarg]

        self.trainer = pl.Trainer(**trainer_kwargs)

    def fit(
        self,
        pipeline: Pipeline,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        vocab_config: Optional[Union[str, VocabularyConfiguration]] = "default",
        lazy: bool = False,
    ):
        """Train the pipeline

        Parameters
        ----------
        pipeline
            Pipeline to train
        train_dataset
            The training dataset
        valid_dataset
            The validation dataset
        vocab_config
            A `VocabularyConfiguration` to create/extend the pipeline's vocabulary.
            If 'default' (str), we will use the default configuration
            `VocabularyConfiguration(datasets=[training_data])`.
            If None, we will leave the pipeline's vocabulary untouched.
        lazy
            If true, instances are lazily loaded from disk, otherwise they are loaded into memory.
        """
        # create vocab
        vocab_config = (
            VocabularyConfiguration(datasets=[train_dataset])
            if vocab_config == "default"
            else vocab_config
        )
        if vocab_config is not None:
            pipeline.create_vocab(vocab_config=vocab_config, lazy=lazy)

        # create dataloaders
        train_dataloader = create_dataloader(
            train_dataset.to_instances(pipeline, lazy=lazy),
            batch_size=self._config.batch_size,
            data_bucketing=self._config.data_bucketing,
        )
        valid_dataloader = (
            create_dataloader(
                valid_dataset.to_instances(pipeline, lazy=lazy),
                batch_size=self._config.batch_size,
                data_bucketing=self._config.data_bucketing,
            )
            if valid_dataset is not None
            else None
        )

        # create optimizer
        pipeline.model.optimizer = Optimizer.from_params(
            Params(
                {
                    "model_parameters": pipeline.model.named_parameters(),
                    **self._config.optimizer,
                }
            )
        )

        self.trainer.fit(
            pipeline.model,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


def create_dataloader(
    instance_dataset: InstancesDataset,
    batch_size: int = 16,
    data_bucketing: bool = False,
) -> PyTorchDataLoader:
    """Returns a pytorch DataLoader for AllenNLP instances

    Parameters
    ----------
    instance_dataset
        The dataset of instances for the DataLoader

    Returns
    -------
    data_loader
    """
    if data_bucketing and isinstance(instance_dataset, IterableDataset):
        _LOGGER.warning(
            "'data_bucketing' is not supported for lazily loaded data. We will deactivate it."
        )
        data_bucketing = False

    return PyTorchDataLoader(
        instance_dataset,
        batch_size=1 if data_bucketing else self._config.batch_size,
        batch_sampler=BucketBatchSampler(
            data_source=instance_dataset,
            batch_size=self._config.batch_size,
        )
        if data_bucketing
        else None,
    )
