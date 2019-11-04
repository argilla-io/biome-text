import logging

from allennlp.training import CallbackTrainer
from allennlp.training.callbacks import Callback, Events, handle_event


@Callback.register("logging")
class LoggingCallback(Callback):
    """
    This callbacks allows controls the logging messages during the training process
    """

    __LOGGER = logging.getLogger(__name__)

    @handle_event(Events.TRAINING_START)
    def on_training_starts(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Training starts...")

    @handle_event(Events.TRAINING_END)
    def on_training_ends(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Training ends...")

    @handle_event(Events.BATCH_START)
    def on_batch_starts(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Batch starts...")

    @handle_event(Events.BATCH_END)
    def on_batch_ends(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Batch starts...")

    @handle_event(Events.FORWARD)
    def on_forward(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Forward")

    @handle_event(Events.BACKWARD)
    def on_backward(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Backward")

    @handle_event(Events.VALIDATE)
    def on_validate(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Validate")

    @handle_event(Events.ERROR)
    def on_error(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Error")

    @handle_event(Events.EPOCH_START)
    def on_epoch_starts(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Epoch starts...")

    @handle_event(Events.EPOCH_END)
    def on_epoch_end(self, trainer: CallbackTrainer):
        self.__LOGGER.debug("Epoch starts...")
