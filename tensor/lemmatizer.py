from functools import lru_cache

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')


@lru_cache()
def get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def lemmatize(word) -> str:
    return get_lemmatizer().lemmatize(word)
