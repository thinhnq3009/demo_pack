# training_module
import json
import pickle
import random

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

from exception.no_answer_exception import NoAnswerException
from schema.base_chat_bot import BaseChatBot

lemmatizer = WordNetLemmatizer()


class ModelRunner(BaseChatBot):
    # loading the files we made previously
    # intents = json.loads(open("../intents.json").read())
    # words = pickle.load(open('traning.pkl', 'rb'))
    # classes = pickle.load(open('classes.pkl', 'rb'))
    # model = load_model('my_model.keras')
    # ERROR_THRESHOLD: float = 0.5

    def __init__(self, path: str, error_threshold: float = 0.5):
        if path is None:
            raise Exception("Path not found !!")
        self.model = load_model(path + "_model.keras")
        self.words = pickle.load(open(path + '_words.pkl', 'rb'))
        self.classes = pickle.load(open(path + "_classes.pkl", 'rb'))
        self.intents = json.loads(open(path + "_dataset.json").read())
        self.ERROR_THRESHOLD: float = error_threshold

    @staticmethod
    def clean_up_sentences(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word)
                          for word in sentence_words]
        return sentence_words

    def bag_word(self, sentence):
        # separate out words from the input sentence
        sentence_words = ModelRunner.clean_up_sentences(sentence)
        if self.words is None:
            pass
            # TODO init
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):

                # check whether the word
                # is present in the input as well
                if word == w:
                    # as the list of words
                    # created earlier.
                    bag[i] = 1

        # return a numpy array
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_word(sentence)
        res = self.model.predict(np.array([bow]))[0]
        results = [[i, r] for i, r in enumerate(res)
                   if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]],
                                'probability': str(r[1])})
        return return_list

    def get_responses(self, intents_list):
        if len(intents_list) == 0:
            raise NoAnswerException("No answer found !!")

        results = []
        for intent in intents_list:
            tag = intent['intent']
            list_of_intents = self.intents
            for i in list_of_intents:
                if i['tag'] == tag:
                    # prints a random response
                    result = random.choice(i['responses'])
                    results.append(result)
                    break
        return results

    def get_response(self, message: str | dict | list[dict]) -> list[str] | str:
        ints = self.predict_class(message)
        res = self.get_responses(ints)
        return res
