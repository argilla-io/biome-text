import argparse
import inspect
import logging

from allennlp.commands import Subcommand
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, DataIterator
from allennlp.data.tokenizers.word_filter import WordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer
from allennlp.models import Model
from allennlp.modules import (
    Seq2VecEncoder,
    TextFieldEmbedder,
    TokenEmbedder,
    Seq2SeqEncoder,
    SimilarityFunction,
)
from allennlp.modules.elmo import Elmo
from allennlp.nn import Activation, Initializer
from allennlp.nn.regularizers import Regularizer
from allennlp.predictors import Predictor
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metrics import Metric
from allennlp.training.optimizers import Optimizer

__logger = logging.getLogger(__name__)

__REGISTRABLE_MAP = {
    "Predictor": Predictor,
    "WordSplitter": WordSplitter,
    "TokenIndexer": TokenIndexer,
    "Metric": Metric,
    "WordStemmer": WordStemmer,
    "WordFilter": WordFilter,
    "TextFieldEmbedder": TextFieldEmbedder,
    "DatasetReader": DatasetReader,
    "LearningRateScheduler": LearningRateScheduler,
    "Tokenizer": Tokenizer,
    "TokenEmbedder": TokenEmbedder,
    "Optimizer": Optimizer,
    "Elmo": Elmo,
    "Activation": Activation,
    "Regularizer": Regularizer,
    "DataIterator": DataIterator,
    "Initializer": Initializer,
    "Seq2SeqEncoder": Seq2SeqEncoder,
    "Seq2VecEncoder": Seq2VecEncoder,
    "SimilarityFunction": SimilarityFunction,
    "Model": Model,
}


class DescribeRegistrable(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Describe expected params for a registrable component."""

        subparser = parser.add_parser(
            name, description=description, help="Describe a registrable"
        )
        subparser.add_argument(
            "-t",
            "--type",
            type=str,
            default="",
            help="The base class for the described name component (predictor, model, encoder or reader)",
        )
        subparser.add_argument(
            "-n", "--name", type=str, default="", help="the registered named"
        )

        subparser.set_defaults(func=describe_registrable)

        return subparser


def describe_registrable(args: argparse.Namespace) -> None:
    if not args.type:
        logging.warning(
            "No type provide. Select one of [%s]",
            ", ".join(["'%s'" % key for key, _ in __REGISTRABLE_MAP.items()]),
        )
        exit(1)

    registrable_type = __REGISTRABLE_MAP[args.type]
    if not args.name:
        logging.warning(
            "No name provided. Select an available name from [%s]",
            registrable_type.list_available(),
        )
    else:
        registrable = registrable_type.by_name(args.name)
        init_signature = inspect.signature(registrable.__init__)
        __logger.info(init_signature.parameters)
