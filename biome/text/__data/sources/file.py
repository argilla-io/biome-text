import glob
import logging
import os
from typing import List, Dict, Any, Tuple, Optional

import dask.bag as db
import dask.dataframe as df
import pandas as pd
from biome.text.data.readers import pdf
from biome.text.data.readers import DocumentBlock
from dask.bag import Bag

__logger = logging.getLogger(__name__)

ID = 'id'
RESOURCE = 'resource'

__DASK_PATH_COLUMN_NAME = 'path'


def __row2dict(row: Tuple, columns: List[str], default_path: Optional[str] = None) -> Dict[str, Any]:
    id = row[0]
    data = row[1:]

    # For duplicated column names, pandas append a index prefix with dots '.' We prevent
    # index failures by replacing for '_'
    sanitized_columns = [column.replace('.', '_') for column in columns]
    data = dict([(ID, id)] + list(zip(sanitized_columns, data)))

    # DataFrame.read_csv allows include path column called `path`
    data[RESOURCE] = data.get(__DASK_PATH_COLUMN_NAME, str(default_path))

    return data


def from_csv(path: str, columns: List[str] = [], **params) -> Bag:
    dataframe = df.read_csv(path, **params, include_path_column=True)

    columns = [str(column).strip() for column in dataframe.columns] if not columns else columns
    return dataframe \
        .to_bag(index=True) \
        .map(__row2dict, columns)


def from_json(path: str, **params) -> Bag:
    dataframe = df.read_json(path, **params)

    columns = [str(column).strip() for column in dataframe.columns]
    return dataframe \
        .to_bag(index=True) \
        .map(__row2dict, columns, path)


def from_excel(path: str, **params) -> Bag:
    file_names = glob.glob(path, recursive=True)
    dataframe = df.from_pandas(
        pd.read_excel(path, **params).fillna(''), npartitions=max(1, len(file_names))
    )

    columns = [str(column).strip() for column in dataframe.columns]

    return dataframe \
        .to_bag(index=True) \
        .map(__row2dict, columns, path)


def from_documents(path: str,
                   recursive: bool = False,
                   include_hierarchical_info: bool = False,
                   **kwargs) -> Bag:
    def sanitize_block(block: Dict[str, Any]) -> Dict[str, Any]:

        def del_property_safety(data: Dict, property_name: str) -> None:
            try:
                del data[property_name]
            except KeyError as _:
                pass

        del_property_safety(block, 'parent')
        del_property_safety(block, 'children')

        return block

    supported_formats = {
        '.pdf': pdf.read_blocks
    }

    def non_supported_extension(path: str, **_) -> List:
        __logger.warning('Cannot process file {}. Non supported exension'.format(path))
        return []

    def process_file(file_path: str, **kwargs) -> List[pdf.DocumentBlock]:
        _, extension = os.path.splitext(file_path)
        processor = supported_formats.get(extension, non_supported_extension)

        file_blocks = processor(file_path, **kwargs)
        return file_blocks

    file_names = glob.glob(path, recursive=recursive)
    # TODO assert has files
    ds: Bag[DocumentBlock] = db.from_sequence(file_names) \
        .map(process_file, **kwargs) \
        .flatten()

    return ds \
        if include_hierarchical_info \
        else ds.map(sanitize_block)
