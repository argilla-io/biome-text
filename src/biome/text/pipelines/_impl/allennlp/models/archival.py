import gzip
import os
import shutil
import tempfile

import smart_open
from allennlp.models import load_archive as _load_archive, Archive


def load_archive(
    archive_file: str,
    cuda_device: int = -1,
    overrides: str = "",
    weights_file: str = None,
) -> Archive:
    new_location = to_local_archive(archive_file)
    return _load_archive(new_location, cuda_device, overrides, weights_file)


def to_local_archive(archive_file: str) -> str:
    """
    Wraps archive download to support remote locations (s3, hdfs,...)
    """
    tempdir = tempfile.mkdtemp()
    local_file = os.path.join(tempdir, "model.tar.gz")
    with smart_open.open(archive_file, mode="rb") as archive, gzip.open(
        local_file, "wb"
    ) as output:
        shutil.copyfileobj(archive, output, length=-1)
    return local_file
