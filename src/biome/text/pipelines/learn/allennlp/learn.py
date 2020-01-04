import glob
import logging
import os
import shutil
from typing import Optional

from allennlp.commands.fine_tune import fine_tune_model
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from allennlp.models.archival import CONFIG_NAME

from biome.text.models import load_archive
from biome.text.pipelines.learn.allennlp.defs import BiomeConfig

__LOGGER = logging.getLogger(__name__)  # pylint: disable=invalid-name


def learn(
    output: str,
    model_spec: Optional[str] = None,
    model_binary: Optional[str] = None,
    vocab: Optional[str] = None,
    trainer_path: str = "",
    train_cfg: str = "",
    validation_cfg: str = "",
    test_cfg: Optional[str] = None,
    verbose: bool = False,
) -> Model:

    if verbose:
        logging.getLogger("allennlp").setLevel(logging.INFO)
    else:
        logging.getLogger("allennlp").setLevel(logging.WARNING)

    __LOGGER.info("Starting up learning process.")
    if not model_binary and not model_spec:
        raise ConfigurationError("Missing parameter --spec/--binary")

    allennlp_configuration = BiomeConfig(
        model_path=model_spec,
        trainer_path=trainer_path,
        vocab_path=vocab,
        train_path=train_cfg,
        validation_path=validation_cfg,
        test_path=test_cfg,
    ).to_allennlp_params()

    # Vocabulary is needed for components instantiation
    # TODO: Include a proper checking of the model configuration
    # _logger.info("Checking model configuration")
    # check_model_configuration(Params(deepcopy(allennlp_configuration)))

    allennlp_configuration = allennlp_configuration.copy()
    if model_binary:
        archive = load_archive(model_binary)
        __LOGGER.info(
            f"Loading '{BiomeConfig.MODEL_FIELD}' config: "
            f"{archive.config.as_dict()[BiomeConfig.MODEL_FIELD]}"
        )
        __LOGGER.info(
            f"Loading '{BiomeConfig.DATASET_READER_FIELD}' config:"
            f"{archive.config.as_dict()[BiomeConfig.DATASET_READER_FIELD]}"
        )
        __LOGGER.info(f"Provided configs: {allennlp_configuration}")
        # The model params are ignored by the `fine_tune_model` method
        fine_tune_params = Params(
            {
                BiomeConfig.DATASET_READER_FIELD: archive.config.get(
                    BiomeConfig.DATASET_READER_FIELD
                ).as_dict(),
                BiomeConfig.MODEL_FIELD: archive.config.get(
                    BiomeConfig.MODEL_FIELD
                ).as_dict(),
                **allennlp_configuration,
            }
        )
        # Force clean folder for run fine tuning properly
        shutil.rmtree(output, ignore_errors=True)

        return fine_tune_model(
            model=archive.model,
            params=fine_tune_params,
            serialization_dir=output,
            extend_vocab=True,
            file_friendly_logging=True,
        )
    else:
        params = Params(allennlp_configuration)
        is_recovered = recover_output_folder(output, params)
        return train_model(
            params=params,
            serialization_dir=output,
            file_friendly_logging=True,
            recover=is_recovered,
        )


def recover_output_folder(output: str, params: Params) -> bool:
    """If output folder already exists, we automatically recover the generated vocab in this folder.

    Allows reuse the generated vocab if something went wrong in previous executions

    Parameters
    ----------
    output
        Path to the output folder
    params
        Parameters for the train command

    Returns
    -------
    is_recovered
        True if existing output folder is recovered, False if output folder does not exist.
    """
    if not os.path.isdir(output):
        return False
    else:
        [
            os.remove(file)
            for pattern in [
                os.path.join(output, "*.th"),
                os.path.join(output, "*.json"),
                os.path.join(output, "**/events.out*"),
            ]
            for file in glob.glob(pattern, recursive=True)
        ]
        params.to_file(os.path.join(output, CONFIG_NAME))
        __LOGGER.warning(
            f"Using vocab from recovered output folder '{output}' if available."
        )

        return True
