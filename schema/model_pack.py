from typing import Any

from keras.models import Sequential
from pydantic import Field


class ModelPack:
    model: Any = Field(..., title="Model")  # Chứa model
    words: list[str] = Field(..., title="Words")  # Chứa các từ đã được học
    classes: list[str] = Field(..., title="Classes")  # Chứa các tag của câu hỏi
    intents: dict = Field(..., title="Intents")  # Chứa tag và câu trả lời

    def __init__(self, model: Sequential, words: list[str], classes: list[str], intents: dict):
        self.model = model
        self.words = words
        self.classes = classes
        self.intents = intents
