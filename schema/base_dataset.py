import json

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    tag: str = Field(..., title='Tag', min_length=1)
    patterns: list[str] = Field(..., title='Patterns')
    responses: list[str] = Field(..., title='Responses')

    def get_tag(self) -> str:
        return self.tag

    def get_patterns(self) -> list[str]:
        return self.patterns

    def get_responses(self) -> list[str]:
        return self.responses


class Dataset:
    def add_item(self, item: DatasetItem):
        self.data.append(item)

    def write(self, path: str):
        json.dump([i.dict() for i in self.data], open(path, 'w'))

    def __init__(self, data: list[DatasetItem] = None):
        self.data = data or []

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __add__(self, other):
        new_data = list(self.data)
        new_data.extend(other.data)
        return Dataset(new_data)
