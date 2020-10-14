import logging
import multiprocessing
import pickle
from typing import Union, Dict, Iterable, List

import datasets
from allennlp.data import AllennlpDataset, AllennlpLazyDataset, Instance

from biome.text import Pipeline

InstancesDataset = Union[AllennlpDataset, AllennlpLazyDataset]


class Dataset:
    """A dataset to be used with biome.text Pipelines

    Basically a light wrapper around HuggingFace's `datasets.Dataset`

    Parameters
    ----------
    dataset
        A HuggingFace `datasets.Dataset`
    """

    _LOGGER = logging.getLogger(__name__)
    _PICKLED_INSTANCES_COL_NAME = "PICKLED_INSTANCES_FOR_BIOME_PIPELINE"

    def __init__(
        self, dataset: datasets.Dataset,
    ):
        self.dataset: datasets.Dataset = dataset

    @classmethod
    def load_dataset(cls, *args, split, **kwargs):
        return cls(datasets.load_dataset(*args, split=split, **kwargs))

    @classmethod
    def read_json(cls, paths: Union[str, List[str]]):
        return cls.load_dataset("json", data_file=paths, split="train")

    @classmethod
    def read_csv(cls, paths: Union[str, List[str]]):
        return cls.load_dataset("csv", data_file=paths, split="train")

    def to_instances(self, pipeline: Pipeline, lazy=True) -> InstancesDataset:
        """Convert input to instances for the pipeline

        Parameters
        ----------
        pipeline
            The pipeline for which to create the instances.
        lazy
            If true, instanes are lazily read from disk, otherwise they are kept in memory.
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
            num_proc=min(
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
