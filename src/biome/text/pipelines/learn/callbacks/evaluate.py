import logging
import os

from allennlp.common import Params
from allennlp.common.checks import parse_cuda_device
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import CallbackTrainer
from allennlp.training.callbacks import Callback, handle_event, Events
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.util import evaluate


@Callback.register("evaluate")
class EvaluateCallback(Callback):
    """
    This callback allows to a callback trainer evaluate the model against a test dataset

    Attributes
    ----------
    serialization_dir:str
        The experiment folder
    """

    _LOGGER = logging.getLogger(__name__)

    def __init__(self, serialization_dir: str):
        params = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME))
        pieces = TrainerPieces.from_params(
            params, serialization_dir, recover=True
        )  # pylint: disable=no-member

        self._evaluation_dataset = pieces.test_dataset
        self._evaluation_iterator = pieces.validation_iterator or pieces.iterator
        self._cuda_device = parse_cuda_device(params.pop("cuda_device", -1))

    @handle_event(Events.TRAINING_END, priority=100)
    def evaluate_dataset(self, trainer: CallbackTrainer) -> None:
        """
        This method launches an test dataset (if defined) evaluation when the training ends
        and adds the test metrics to trainer metrics before they are processed (thanks to priority argument)

        Parameters
        ----------
        trainer:CallbackTrainer
            The main callback trainer
        """
        if not self._evaluation_dataset:
            self._LOGGER.warning("No test data found")
            return

        test_metrics = evaluate(
            trainer.model,
            self._evaluation_dataset,
            self._evaluation_iterator,
            cuda_device=self._cuda_device,  # pylint: disable=protected-access,
            batch_weight_key="",
        )

        for key, value in test_metrics.items():
            trainer.metrics["test_" + key] = value
