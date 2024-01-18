from abc import abstractmethod

from schema.base_dataset import Dataset


class BaseDatasetReader:

    @abstractmethod
    def read(self) -> Dataset:
        pass
