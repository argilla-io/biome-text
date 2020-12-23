import copy
import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

import uvicorn
from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.common.util import sanitize
from allennlp.data import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.models import archive_model
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import GradientDescentTrainer
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from fastapi import FastAPI
from torch.utils.data import IterableDataset

from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import helpers
from biome.text._model import PipelineModel
from biome.text.dataset import InstancesDataset
from biome.text.errors import http_error_handling

_LOGGER = logging.getLogger(__name__)


def _serve(pipeline: Pipeline, port: int):
    """Serves an pipeline as rest api"""

    def make_app() -> FastAPI:
        app = FastAPI()

        @app.post("/predict")
        async def predict(inputs: Dict[str, Any]):
            with http_error_handling():
                return sanitize(pipeline.predict(**inputs))

        @app.post("/explain")
        async def explain(inputs: Dict[str, Any]):
            with http_error_handling():
                return sanitize(pipeline.explain(**inputs))

        @app.get("/_config")
        async def config():
            with http_error_handling():
                return pipeline.config.as_dict()

        @app.get("/_status")
        async def status():
            with http_error_handling():
                return {"ok": True}

        return app

    uvicorn.run(make_app(), host="0.0.0.0", port=port)


class PipelineTrainer:
    """
    Default trainer for `PipelineModel`

    Parameters
    ----------
    pipeline
        The trainable pipeline
    trainer_config
        The trainer configuration
    output_dir
        The used training folder
    training
        The training instances dataset
    validation
        The validation instances dataset. Optional
    test
        The test instances dataset. Optional
    batch_weight_key
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    epoch_callbacks
        A list of callbacks that will be called at the end of every epoch, and at the start of
        training (with epoch = -1).
    """

    __LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        pipeline: Pipeline,
        trainer_config: TrainerConfiguration,
        output_dir: str,
        training: InstancesDataset,
        validation: Optional[InstancesDataset] = None,
        test: Optional[InstancesDataset] = None,
        batch_weight_key: str = "",
        epoch_callbacks: List["allennlp.training.EpochCallback"] = None,
    ):
        self._pipeline = pipeline
        self._trainer_config = copy.deepcopy(trainer_config)
        self._output_dir = output_dir
        self._batch_weight_key = batch_weight_key
        self._training = training
        self._validation = validation
        self._test = test
        self._epoch_callbacks = epoch_callbacks

        self._setup()

    def _setup(self):
        """Setup the trainer components and local resources"""
        prepare_environment(
            Params(
                {}
                if self._trainer_config.random_seed is None
                else {
                    "random_seed": self._trainer_config.random_seed,
                    "numpy_seed": self._trainer_config.random_seed,
                    "pytorch_seed": self._trainer_config.random_seed,
                }
            )
        )
        os.makedirs(self._output_dir, exist_ok=True)

        serialization_params = sanitize(self._allennlp_configuration())
        with open(os.path.join(self._output_dir, CONFIG_NAME), "w") as param_file:
            json.dump(serialization_params, param_file, indent=4)

        self._pipeline.vocab.save_to_files(os.path.join(self._output_dir, "vocabulary"))

        for dataset in [self._training, self._validation, self._test]:
            if dataset is not None:
                dataset.index_with(self._pipeline.backbone.vocab)

        trainer_params = Params(
            helpers.sanitize_for_params(self._trainer_config.to_allennlp_trainer())
        )

        pipeline_model = self._pipeline._model

        training_data_loader = create_dataloader(
            self._training,
            self._trainer_config.batch_size,
            self._trainer_config.data_bucketing,
            self._trainer_config.batches_per_epoch,
        )

        validation_data_loader = (
            create_dataloader(
                self._validation,
                self._trainer_config.batch_size,
                self._trainer_config.data_bucketing,
            )
            if self._validation
            else None
        )

        self._trainer = Trainer.from_params(
            model=pipeline_model,
            serialization_dir=self._output_dir,
            data_loader=training_data_loader,
            validation_data_loader=validation_data_loader,
            params=trainer_params,
            epoch_callbacks=self._epoch_callbacks,
        )

    def test_evaluation(self) -> Dict[str, Any]:
        """
        Evaluates the model against the test dataset (if defined)

        Returns
        -------
        Test metrics information

        """
        test_data = self._test
        if not test_data:
            return {}

        self.__LOGGER.info("The model will be evaluated using the best epoch weights.")
        return evaluate(
            self._pipeline._model,
            data_loader=PyTorchDataLoader(
                test_data, batch_size=self._trainer_config.batch_size
            ),
            cuda_device=self._trainer.cuda_device,
            batch_weight_key=self._batch_weight_key,
        )

    def train(self) -> Tuple[str, Dict[str, Any]]:
        """
        Train the inner model with given configuration on initialization

        Returns
        -------
        A tuple with trained model path and related metrics information
        """

        from allennlp.models.model import _DEFAULT_WEIGHTS

        try:
            metrics = self._trainer.train()
        except KeyboardInterrupt:
            # if we have completed an epoch, try to create a model archive.
            if os.path.exists(os.path.join(self._output_dir, _DEFAULT_WEIGHTS)):
                logging.info(
                    "Fine-tuning interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                self.save_best_model()
            raise

        for k, v in self.test_evaluation().items():
            metrics["test_" + k] = v

        self.save_best_model()

        with open(os.path.join(self._output_dir, "metrics.json"), "w") as metrics_file:
            metrics_json = json.dumps(metrics, indent=2)
            metrics_file.write(metrics_json)

        return os.path.join(self._output_dir, "model.tar.gz"), metrics

    def save_best_model(self):
        """Packages the best model as tar.gz archive"""
        archive_model(self._output_dir)

    def _allennlp_configuration(self) -> Dict[str, Any]:
        """Creates an allennlp configuration for pipeline train experiment configuration"""
        allennlp_config = {
            # TODO: this does not make much sense, we should save our trainer configuration format!
            "trainer": self._trainer_config.to_allennlp_trainer(),
            "model": {
                "config": self._pipeline.config.as_dict(),
                "type": PipelineModel.__name__,
            },
        }

        return copy.deepcopy({k: v for k, v in allennlp_config.items() if v})


def create_dataloader(
    dataset: InstancesDataset,
    batch_size: int,
    data_bucketing: bool = False,
    batches_per_epoch: Optional[int] = None,
) -> PyTorchDataLoader:
    """Returns a pytorch DataLoader for AllenNLP

    Parameters
    ----------
    dataset
        The data set for the DataLoader
    batch_size
        Size of the batch.
    data_bucketing
        If enabled, try to apply data bucketing over training batches.
    batches_per_epoch
        Determines the number of batches after which an epoch ends.
        If the number is smaller than the total amount of batches in your data,
        the second "epoch" will take off where the first "epoch" ended.
        If this is `None`, then an epoch is set to be one full pass through your data.

    Returns
    -------
    data_loader
    """
    return (
        PyTorchDataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(
                data_source=dataset, batch_size=batch_size
            ),
            batches_per_epoch=batches_per_epoch,
        )
        if data_bucketing and not isinstance(dataset, IterableDataset)
        else PyTorchDataLoader(
            dataset, batch_size=batch_size, batches_per_epoch=batches_per_epoch
        )
    )


def create_trainer_for_finding_lr(
    model: PipelineModel,
    trainer_config: TrainerConfiguration,
    training_data: InstancesDataset,
) -> GradientDescentTrainer:
    """Returns an AllenNLP Trainer used for the learning rate scan.

    Parameters
    ----------
    model
        The underlying model
    trainer_config
        A trainer configuration
    training_data
        The training data
    """
    prepare_environment(Params({}))

    trainer_params = Params(
        helpers.sanitize_for_params(trainer_config.to_allennlp_trainer())
    )

    training_data_loader = create_dataloader(
        training_data, trainer_config.batch_size, trainer_config.data_bucketing
    )

    return cast(
        "GradientDescentTrainer",
        Trainer.from_params(
            model=model,
            data_loader=training_data_loader,
            params=trainer_params,
            serialization_dir=None,
        ),
    )
