from abc import abstractmethod


class BaseChatBot:

    @abstractmethod
    def get_response(self, message: str | dict | list[dict]) -> list[str] | str:
        pass

    def get_name(self) -> str:
        raise NotImplementedError("get_name() is not implemented")
