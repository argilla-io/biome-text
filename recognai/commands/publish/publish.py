import argparse
import os
import logging
from allennlp.commands import Subcommand

# have all the variables populated which are required below

logger = logging.getLogger(__name__)

try:
    import boto3
    import boto3.s3.transfer as boto3_s3
except Exception:
    logger.warning("boto3 not found")

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')


class PublishModel(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Uploads the configuration model (as tar.gz file) to a s3 bucket.'''
        subparser = parser.add_parser(name, description=description, help='Publish model binaries & config to a S3 bucket')

        subparser.add_argument('--model-name', type=str, required=True)
        subparser.add_argument('--model-location', type=str, required=True)
        subparser.add_argument('--version', type=str, required=True)

        subparser.add_argument('--s3-access-key-id', type=str, default=AWS_ACCESS_KEY_ID)
        subparser.add_argument('--s3-secret-access-key', type=str, default=AWS_SECRET_ACCESS_KEY)
        subparser.add_argument('--s3-bucket', type=str, default='models.recogn.ai')

        subparser.set_defaults(func=_publish)

        return subparser


def _publish(args: argparse.Namespace) -> None:
    client = boto3.client('s3', aws_access_key_id=args.s3_access_key_id,
                          aws_secret_access_key=args.s3_secret_access_key)

    filepath = '%s.%s.tar.gz' % (args.model_name, args.version)

    transfer = boto3_s3.S3Transfer(client)
    transfer.upload_file(args.model_location, args.s3_bucket, filepath)
    client.put_object_acl(ACL='public-read', Bucket=args.s3_bucket, Key=filepath)
