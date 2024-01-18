import json

from schema.base_dataset import Dataset, DatasetItem
from schema.reader.base_reader import BaseDatasetReader


class JsonReader(BaseDatasetReader):

    def __init__(self, path: str | list[str] = None):
        self.path = path

    @staticmethod
    def load_data(path: str, encoding: str = "utf-8") -> Dataset:
        data = json.loads(open(path, encoding=encoding).read())
        dataset = Dataset()
        for item in data:
            dataset.add_item(DatasetItem(**dict(item)))
        return dataset

    def read(self) -> Dataset:
        if isinstance(self.path, list):
            dataset = Dataset()
            for path in self.path:
                dataset += self.load_data(path)
            return dataset
        return self.load_data(self.path)
