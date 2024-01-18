class WordBag:

    def __init__(self, list_word=None):
        self.list_word = [] if list_word is None else list_word

    def add_word(self, word: str):
        if word not in self.list_word:
            self.list_word.append(word)
            return True
        return False

    def add_words(self, words: list[str]) -> list[bool]:
        return [self.add_word(word) for word in words]

    def __contains__(self, item):
        return item in self.list_word

    def __iter__(self):
        return iter(self.list_word)

    def __getitem__(self, index):
        return self.list_word[index]

