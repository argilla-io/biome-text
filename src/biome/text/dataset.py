import contextlib
import functools
import inspect
import logging
import multiprocessing
import pickle
from typing import Union, Dict, Iterable, List, Any, Tuple, Optional, TYPE_CHECKING

import datasets
from allennlp.data import AllennlpDataset, AllennlpLazyDataset, Instance

if TYPE_CHECKING:
    from biome.text.pipeline import Pipeline

InstancesDataset = Union[AllennlpDataset, AllennlpLazyDataset]


def copy_sign_and_docs(org_func):
    """Copy the signature and the docstring from the org_func"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(org_func)
        wrapper.__doc__ = org_func.__doc__

        return wrapper

    return decorator


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
    _PICKLED_INSTANCES_COL_NAME = "PICKLED_INSTANCES_FOR_BIOME_PIPELINE"

    def __init__(
        self, dataset: datasets.Dataset,
    ):
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
        """Convenient method to create a Dataset from a json file

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
        """Convenient method to create a Dataset from a csv file

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
    @copy_sign_and_docs(datasets.Dataset.load_from_disk)
    def load_from_disk(cls, *args, **kwargs) -> "Dataset":
        """
        https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.load_from_disk
        """
        return cls(datasets.load_from_disk(*args, **kwargs))

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

    def to_instances(self, pipeline: "Pipeline", lazy=True, num_proc: Optional[int] = None) -> InstancesDataset:
        """Convert input to instances for the pipeline

        Parameters
        ----------
        pipeline
            The pipeline for which to create the instances.
        lazy
            If true, instances are lazily read from disk, otherwise they are kept in memory.
        num_proc
            Number of processes to be spawn. If None, we try to figure out a decent default.
        """
        self._LOGGER.info("Creating instances ...")

        input_columns = [col for col in pipeline.inputs + [pipeline.output] if col]
        dataset_with_pickled_instances = self.dataset.map(
            self._create_and_pickle_instances,
            fn_kwargs={
                "input_columns": input_columns,
                "featurize": pipeline.head.featurize,
                "instances_col_name": self._PICKLED_INSTANCES_COL_NAME,
            },
            # trying to be smart about multiprocessing,
            # at least 1000 examples per process to avoid overhead,
            # but 1000 is a pretty random number, can surely be optimized
            num_proc=num_proc or min(
                max(1, int(len(self.dataset) / 1000)),
                int(multiprocessing.cpu_count() / 2),
            ),
        )

        if lazy:
            return AllennlpLazyDataset(
                instance_generator=self._build_instance_generator(
                    dataset_with_pickled_instances
                ),
                file_path=self._PICKLED_INSTANCES_COL_NAME,
            )

        return AllennlpDataset(
            list(
                self._build_instance_generator(dataset_with_pickled_instances)(
                    self._PICKLED_INSTANCES_COL_NAME
                )
            )
        )

    @staticmethod
    def _create_and_pickle_instances(row, input_columns, featurize, instances_col_name):
        """Helper function to be used together with the `datasets.Dataset.map` method"""
        instance = featurize(**{key: row[key] for key in input_columns})

        return {instances_col_name: pickle.dumps(instance)}

    @staticmethod
    def _build_instance_generator(pickled_instances: datasets.Dataset):
        """Helper function to be used together with the `allennlp.data.AllennlpLazyDataset`"""

        def instance_unpickler(instances_col_name: str) -> Iterable[Instance]:
            for row in pickled_instances:
                yield pickle.loads(row[instances_col_name])

        return instance_unpickler

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
