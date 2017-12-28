import argparse
import logging
from typing import Dict, Any

from allennlp.commands import Subcommand
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

from .kafkaPipelineProcess import KafkaPipelineProcess
from .serviceInstanceConfigServer import ServiceInstanceConfigServer, RedisClient
from allennlp_extensions.commands.utils import yaml_to_dict

_logger = logging.getLogger(__name__)


class KafkaPipelineCommand(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''TODO'''
        subparser = parser.add_parser(name, description=description, help='TODO')

        subparser.add_argument('--config', type=str, default='./kafka.config.yml')

        subparser.set_defaults(func=_serve)

        return subparser


def _load_model_from_params(params: Dict) -> Predictor:
    archive = load_archive(params['location'], params.get('gpu', -1))
    return Predictor.from_archive(archive, params['name'])


def _transform(model: Predictor, redis_client: RedisClient):
    def contains_selector(data):
        selectors = redis_client.get_values()
        # TODO check selector from message
        return True

    def transform_inner(data: Any):
        if contains_selector(data):
            return model.predict_json(data)

    return transform_inner


def _serve(args: argparse.Namespace) -> None:
    config = yaml_to_dict(args.config)

    model = _load_model_from_params(config.pop('model', {}))

    pipeline: KafkaPipelineProcess = KafkaPipelineProcess.from_params(config.pop('kafka', {}))

    config_server: ServiceInstanceConfigServer = ServiceInstanceConfigServer.from_params(config.pop('serve', {}),
                                                                                         service_group=pipeline.group)

    pipeline.set_transformation(_transform(model, config_server.redis_client))

    pipeline.start()  # Launch a separate process
    config_server.start()  # Block until main process end
    pipeline.stop()
