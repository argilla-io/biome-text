from typing import Union

from allennlp.data import AllennlpDataset, AllennlpLazyDataset

from .datasource import DataSource


InstancesDataset = Union[AllennlpDataset, AllennlpLazyDataset]
