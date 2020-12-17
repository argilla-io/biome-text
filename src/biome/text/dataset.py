import contextlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import datasets
from allennlp import __version__ as allennlp__version__
from allennlp.data import AllennlpDataset
from allennlp.data import AllennlpLazyDataset
from allennlp.data import Instance
from datasets.fingerprint import Hasher
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from spacy import __version__ as spacy__version__
from tqdm.auto import tqdm

from biome.text import __version__ as biome__version__
from biome.text import helpers
from biome.text.helpers import copy_sign_and_docs

if TYPE_CHECKING:
    import pandas

    from biome.text.pipeline import Pipeline

InstancesDataset = Union[AllennlpDataset, AllennlpLazyDataset]


class Dataset:
    """A dataset to be used with biome.text Pipelines

    Is is a very light wrapper around HuggingFace's awesome `datasets.Dataset`,
    only including a biome.text specific `to_instances` method.

    Most of the `datasets.Dataset` API is exposed and can be looked up in detail here:
    https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset

    Parameters
    ----------
    dataset
        A HuggingFace `datasets.Dataset`

    Attributes
    ----------
    dataset
        The underlying HuggingFace `datasets.Dataset`
    """

    _LOGGER = logging.getLogger(__name__)
    _CACHED_INSTANCE_LIST_EXTENSION = "instance_list"

    def __init__(self, dataset: datasets.Dataset):
        self.dataset: datasets.Dataset = dataset

    @classmethod
    def load_dataset(cls, *args, split, **kwargs) -> "Dataset":
        """Load a dataset using Huggingface's `datasets.load_dataset` method.

        See https://huggingface.co/docs/datasets/loading_datasets.html

        Parameters
        ----------
        split
            See https://huggingface.co/docs/datasets/splits.html
        *args/**kwargs
            Passed on to the `datasets.load_dataset` method

        Returns
        -------
        dataset
        """
        return cls(datasets.load_dataset(*args, split=split, **kwargs))

    @classmethod
    def from_json(cls, paths: Union[str, List[str]], **kwargs) -> "Dataset":
        """Create a Dataset from a json file

        Parameters
        ----------
        paths
            One or several paths to json files
        **kwargs
            Passed on to the `load_dataset` method

        Returns
        -------
        dataset
        """
        return cls.load_dataset("json", data_files=paths, split="train", **kwargs)

    @classmethod
    def from_csv(cls, paths: Union[str, List[str]], **kwargs) -> "Dataset":
        """Create a Dataset from a csv file

        Parameters
        ----------
        paths
            One or several paths to csv files
        **kwargs
            Passed on to the `load_dataset` method

        Returns
        -------
        dataset
        """
        return cls.load_dataset("csv", data_files=paths, split="train", **kwargs)

    @classmethod
    @copy_sign_and_docs(datasets.Dataset.from_pandas)
    def from_pandas(cls, *args, **kwargs):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.from_pandas
        """
        return cls(datasets.Dataset.from_pandas(*args, **kwargs))

    @classmethod
    @copy_sign_and_docs(datasets.Dataset.from_dict)
    def from_dict(cls, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.from_dict
        """
        return cls(datasets.Dataset.from_dict(*args, **kwargs))

    @classmethod
    def from_elasticsearch(
        cls,
        client: Elasticsearch,
        index: str,
        query: Optional[dict] = None,
        fields: List[str] = None,
    ):
        """Create a Dataset from scanned query records in an elasticsearch index

        Parameters
        ----------
        client:
            The elasticsearch client instance
        index:
            The index, index pattern or alias to fetch documents
        query:
            The es query body
        fields:
            Select fields to extract as ds features

        Returns
        -------
        dataset
        """

        def __clean_document__(document: Dict, _fields: List[str] = None) -> Dict:
            source = {
                **{
                    k: document.get(k)
                    for k in ["_id", "_type", "_index", "_score"]
                    if k in document
                },
                **document.get("source", {}),
            }
            if _fields:
                source = {k: source.get(k) for k in _fields}
            return helpers.stringify(source)

        es_query = query or {}

        scanned_docs = [
            __clean_document__(doc, fields)
            for doc in scan(client=client, query=es_query, index=index)
        ]
        if len(scanned_docs) <= 0:  # prevent empty results
            return cls.from_dict({})

        data_dict = {k: [doc.get(k) for doc in scanned_docs] for k in scanned_docs[0]}
        return cls.from_dict(data_dict)

    @classmethod
    def from_datasets(cls, dataset_list: List["Dataset"]) -> "Dataset":
        """Create a single Dataset by concatenating a list of datasets

        Parameters
        ----------
        dataset_list
            Datasets to be concatenated. They must have the same column types.

        Returns
        -------
        dataset
        """
        return cls(datasets.concatenate_datasets([ds.dataset for ds in dataset_list]))

    @classmethod
    @copy_sign_and_docs(datasets.Dataset.load_from_disk)
    def load_from_disk(cls, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.load_from_disk
        """
        return cls(datasets.load_from_disk(*args, **kwargs))

    @copy_sign_and_docs(datasets.Dataset.save_to_disk)
    def save_to_disk(self, dataset_path: str):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.save_to_disk
        """
        self.dataset.save_to_disk(dataset_path=dataset_path)

    @copy_sign_and_docs(datasets.Dataset.select)
    def select(self, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.select
        """
        return Dataset(self.dataset.select(*args, **kwargs))

    @copy_sign_and_docs(datasets.Dataset.map)
    def map(self, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.map
        """
        return Dataset(self.dataset.map(*args, **kwargs))

    @copy_sign_and_docs(datasets.Dataset.filter)
    def filter(self, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.filter
        """
        return Dataset(self.dataset.filter(*args, **kwargs))

    @copy_sign_and_docs(datasets.Dataset.flatten_)
    def flatten_(self, *args, **kwargs):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.flatten_
        """
        self.dataset.flatten_(*args, **kwargs)

    @copy_sign_and_docs(datasets.Dataset.rename_column_)
    def rename_column_(self, *args, **kwargs):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.rename_column_
        """
        self.dataset.rename_column_(*args, **kwargs)

    @copy_sign_and_docs(datasets.Dataset.remove_columns_)
    def remove_columns_(self, *args, **kwargs):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.remove_columns_
        """
        self.dataset.remove_columns_(*args, **kwargs)

    @copy_sign_and_docs(datasets.Dataset.shuffle)
    def shuffle(self, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.shuffle
        """
        return Dataset(self.dataset.shuffle(*args, **kwargs))

    @copy_sign_and_docs(datasets.Dataset.sort)
    def sort(self, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.sort
        """
        return Dataset(self.dataset.sort(*args, **kwargs))

    @copy_sign_and_docs(datasets.Dataset.train_test_split)
    def train_test_split(self, *args, **kwargs) -> Dict[str, "Dataset"]:
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.train_test_split
        """
        dataset_dict = self.dataset.train_test_split(*args, **kwargs)
        return {
            "train": Dataset(dataset_dict["train"]),
            "test": Dataset(dataset_dict["test"]),
        }

    @copy_sign_and_docs(datasets.Dataset.unique)
    def unique(self, *args, **kwargs) -> List[Any]:
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.unique
        """
        return self.dataset.unique(*args, **kwargs)

    @copy_sign_and_docs(datasets.Dataset.set_format)
    def set_format(self, *args, **kwargs):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.set_format
        """
        self.dataset.set_format(*args, **kwargs)

    @copy_sign_and_docs(datasets.Dataset.reset_format)
    def reset_format(self):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.reset_format
        """
        self.dataset.reset_format()

    @contextlib.contextmanager
    @copy_sign_and_docs(datasets.Dataset.formatted_as)
    def formatted_as(self, *args, **kwargs):
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.DatasetDict.formatted_as
        """
        with self.dataset.formatted_as(*args, **kwargs) as internal_cm:
            try:
                yield internal_cm
            finally:
                pass

    @property
    def column_names(self) -> List[str]:
        """Names of the columns in the dataset"""
        return self.dataset.column_names

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the dataset (number of columns, number of rows)"""
        return self.dataset.shape

    @property
    def num_columns(self) -> int:
        """Number of columns in the dataset"""
        return self.dataset.num_columns

    @property
    def num_rows(self) -> int:
        """Number of rows in the dataset (same as `len(dataset)`)"""
        return self.dataset.num_rows

    @property
    def format(self) -> dict:
        return self.dataset.format

    def to_instances(
        self, pipeline: "Pipeline", lazy: bool = False, use_cache: bool = True
    ) -> InstancesDataset:
        """Convert input to instances for the pipeline

        Parameters
        ----------
        pipeline
            The pipeline for which to create the instances.
        lazy
            If true, instances are lazily loaded from disk, otherwise they are loaded into memory.
        use_cache
            If true, we will try to reuse cached instances. Ignored when `lazy=True`.

        Returns
        -------
        instance_dataset
        """
        input_columns = [(col, False) for col in pipeline.inputs if col]
        input_columns.extend([(col, True) for col in pipeline.output if col])

        if lazy:
            return AllennlpLazyDataset(
                instance_generator=self._build_instance_generator(
                    pipeline, self.dataset, input_columns
                ),
                file_path="dummy",
            )

        fingerprint = self._create_fingerprint_for_instance_list(pipeline)
        instance_list = self._load_instance_list(fingerprint) if use_cache else None
        if instance_list is None:
            instance_generator = self._build_instance_generator(
                pipeline, self.dataset, input_columns
            )("dummy")
            tqdm_prog = tqdm(
                instance_generator,
                desc="Loading instances into memory",
                total=len(self.dataset),
            )
            instance_list = [instance for instance in tqdm_prog]
            self._cache_instance_list(instance_list, fingerprint)

        return AllennlpDataset(instance_list)

    @staticmethod
    def _build_instance_generator(
        pipeline: "Pipeline",
        dataset: datasets.Dataset,
        input_columns: List[Tuple[str, bool]],
    ) -> Callable[[str], Iterable[Instance]]:
        """Build the instance generator

        Parameters
        ----------
        pipeline
            The pipeline for which the instances are created for
        dataset
            The dataset underlying the instances
        input_columns
            The columns in the dataset used for the instances

        Returns
        -------
        instance_generator
        """
        # we need a dummy str to comply with AllennlpLazyDataset API
        def instance_generator(dummy: str) -> Iterable[Instance]:
            for row in dataset:
                instance = pipeline.head.featurize(
                    **{
                        key: row.get(key) if optional else row[key]
                        for key, optional in input_columns
                    }
                )
                # We skip examples for which the head could not create an instance
                # We leave it to the head to issue a logging.warning for these examples
                if instance is not None:
                    yield instance

        return instance_generator

    def _create_fingerprint_for_instance_list(self, pipeline: "Pipeline") -> str:
        """Create a fingerprint for the instance list

        The fingerprint is based on:
        - the fingerprint of the previous dataset
        - the tokenizer config
        - the indexer config of the features
        - the biome__version__, allennlp__version__ and spaCy__version__ just to be completely sure!

        Parameters
        ----------
        pipeline
            Pipeline with the tokenizer and indexer config of the features

        Returns
        -------
        fingerprint
            String of hexadecimal digits
        """
        hasher = Hasher()
        hasher.update(self.dataset._fingerprint)  # necessary evil ...
        hasher.update(vars(pipeline.backbone.tokenizer.config))
        for feature in pipeline.config.features:
            hasher.update(feature.config["indexer"])
        hasher.update(biome__version__)
        hasher.update(allennlp__version__)
        hasher.update(spacy__version__)

        return hasher.hexdigest()

    def _load_instance_list(self, fingerprint: str) -> Optional[List[Instance]]:
        """Load the cached instance list

        Parameters
        ----------
        fingerprint
            Fingerprint of the instance list

        Returns
        -------
        instance_list
            Returns None, if no cached instances are found
        """
        try:
            cache_path = (
                Path(self.dataset.cache_files[0]["filename"]).parent
                / f"{fingerprint}.{self._CACHED_INSTANCE_LIST_EXTENSION}"
            )
            with cache_path.open("rb") as file:
                self._LOGGER.warning(f"Reusing cached instances ({cache_path})")
                instance_list = pickle.load(file)
        except (IndexError, KeyError, FileNotFoundError):
            return None

        return instance_list

    def _cache_instance_list(self, instance_list: List[Instance], fingerprint: str):
        """Cache an instance list

        Parameters
        ----------
        instance_list
            List of instances to be cached
        fingerprint
            Fingerprint of the instance list
        """
        try:
            cache_path = (
                Path(self.dataset.cache_files[0]["filename"]).parent
                / f"{fingerprint}.{self._CACHED_INSTANCE_LIST_EXTENSION}"
            )
            with cache_path.open("wb") as file:
                self._LOGGER.info(f"Caching instances to {cache_path})")
                pickle.dump(instance_list, file)
        except (IndexError, KeyError, FileNotFoundError):
            pass

    @copy_sign_and_docs(datasets.Dataset.cleanup_cache_files)
    def cleanup_cache_files(self):
        # Apart from cleaning up datasets cache files, this also checks if there is actually a cache dir
        nr_of_removed_files = self.dataset.cleanup_cache_files()
        if nr_of_removed_files is not None:
            cache_dir_path = Path(self.dataset.cache_files[0]["filename"]).parent
            # cleaning up the cached instance lists
            for file_path in cache_dir_path.glob(
                f"*.{self._CACHED_INSTANCE_LIST_EXTENSION}"
            ):
                file_path.unlink()
                self._LOGGER.info(f"Removed {file_path}")
                nr_of_removed_files += 1

        return nr_of_removed_files

    def head(self, n: Optional[int] = 10) -> "pandas.DataFrame":
        """Return the first n rows of the dataset as a pandas.DataFrame

        Parameters
        ----------
        n
            Number of rows. If None, return the whole dataset as a pandas DataFrame

        Returns
        -------
        dataframe
        """
        with self.dataset.formatted_as("pandas"):
            return cast("pandas.DataFrame", self.dataset[:n])

    def __del__(self):
        self.dataset.__del__()

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Dict, List]:
        return self.dataset.__getitem__(key)

    def __len__(self):
        return self.dataset.__len__()

    def __iter__(self):
        yield from self.dataset.__iter__()

    def __repr__(self):
        return self.dataset.__repr__()
