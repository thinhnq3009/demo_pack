from exception.no_answer_exception import NoAnswerException
from schema.base_chat_bot import BaseChatBot


class ChatbotChain(BaseChatBot):

    def __init__(self, chatbot_list: list[BaseChatBot] = None):
        self.chatbot_list = chatbot_list or []

    def add_chatbot(self, chatbot: BaseChatBot):
        self.chatbot_list.append(chatbot)

    def remove_chatbot(self, chatbot: BaseChatBot):
        self.chatbot_list.remove(chatbot)

    def clear_chatbot(self):
        self.chatbot_list.clear()

    def get_chatbots(self):
        return self.chatbot_list

    def get_chatbot(self, index: int):
        return self.chatbot_list[index]

    def update_chatbot(self, chatbot_list: BaseChatBot, index: int):
        self.chatbot_list[index] = chatbot_list

    def get_chatbot_size(self):
        return len(self.chatbot_list)

    def get_response(self, message: str | dict | list[dict]) -> list[str] | str:
        for chatbot in self.chatbot_list:
            if chatbot is self:
                continue
            try:
                return chatbot.get_response(message)
            except NoAnswerException as e:
                continue
        raise NoAnswerException("No answer found")
