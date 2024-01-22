from abc import abstractmethod


class BaseChatBot:

    @abstractmethod
    def get_response(self, message: str | dict | list[dict]) -> list[str] | str:
        pass
