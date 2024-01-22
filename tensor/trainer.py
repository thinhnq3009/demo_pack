# training_module
# py_training
import pickle
import random
from typing import Any, Optional

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from pydantic import BaseModel, Field

from exception.trainer_not_ready_error import TrainerNotReadyError
from schema.base_dataset import Dataset
from schema.reader.base_reader import BaseDatasetReader
from tensor.lemmatizer import lemmatize
from tensor.word_bag import WordBag


class ModelTrainer(BaseModel):
    name_model: str = Field(..., title="Name of model", min_length=1)

    ignore_letters: list[str] = ["?", "!", ".", ","]

    model: Any = None

    words: Any = WordBag()  # Chứa từ đã được xử lý
    classes: list[str] = []  # Chứa các tag

    documents: list = []  # chứa các từ được cắt ra từ câu hỏi và tag của câu hỏi đó

    data_train: list = []
    classes_train: Optional[list] = None
    words_train: Optional[list] = None

    __is_numerical: bool = False  # Biến này đảm bảo rằng dữ liệu đã được xư lý và chuyển về dạng vector
    __dataset: Dataset = Dataset()

    def read_dateset(self, reader: BaseDatasetReader):
        dataset = reader.read()
        self.__dataset += dataset
        self.load_dataset(dataset)

    def load_dataset(self, dataset: Dataset):
        for dataset_item in dataset:
            for pattern in dataset_item.patterns:
                # cắt các từ trong câu hỏi
                word_list = nltk.word_tokenize(pattern)
                self.words.add_words(word_list)  # and adding them to words list

                self.documents.append((word_list, dataset_item.tag))

            # lưu tag lại
            if dataset_item.tag not in self.classes:
                self.classes.append(dataset_item.tag)

        self.__is_numerical = False

    # Xử lý các từ trong câu hỏi
    def __process_words(self):
        processed = [lemmatize(word) for word in self.words if word not in self.ignore_letters]
        processed = sorted(set(processed))
        self.words = WordBag(list_word=processed)

    #  Số hoá từ và chuyển về dạng vector
    def __numerical(self):
        self.__process_words()
        data_train = []
        output_empty = [0] * len(self.classes)
        for document in self.documents:
            bag_bit = []
            word_patterns = document[0]
            word_patterns = [lemmatize(word.lower()) for word in word_patterns]

            for word in self.words:
                bag_bit.append(1 if word in word_patterns else 0)

            # making a copy of the output_empty
            output_row = list(output_empty)

            # Chuyển tag thành vector
            output_row[self.classes.index(document[1])] = 1
            data_train.append([bag_bit, output_row])
        random.shuffle(data_train)
        data_train = np.array(data_train, dtype=object)

        self.data_train = data_train
        self.words_train = list(data_train[:, 0])
        self.classes_train = list(data_train[:, 1])
        self.__is_numerical = True

    def init_model(self):

        self.__numerical()

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.words_train[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.classes_train[0]), activation='softmax'))

    def set_model(self, model: Sequential):
        self.model = model

    def __check_numerical(self):
        if self.__is_numerical:
            raise TrainerNotReadyError("Data is not numerical, please call numerical()")

    def compile_model(self, epochs: int = 200, batch_size: int = 5, verbose: int = 1, save: bool = True,
                      filename: str = None, path: str = ""):

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])
        hist = self.model.fit(np.array(self.words_train), np.array(self.classes_train),
                              epochs=epochs, batch_size=batch_size, verbose=verbose)

        if save:
            path += f'/{filename or self.name_model}'
            self.model.save(f'{path}_model.keras', hist)
            pickle.dump(self.words, open(f'{path}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{path}_classes.pkl', 'wb'))
            # write self.dataset to file .json
            self.__dataset.write(path + "_dataset.json")

        return self.model, hist
