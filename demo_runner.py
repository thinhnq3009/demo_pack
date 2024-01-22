from exception.no_answer_exception import NoAnswerException
from tensor.runner import ModelRunner

runner = ModelRunner(path="models/gpt-1")

while True:
    try:
        prompt = input("Prompt:")
        res = runner.get_response(prompt)
        print(res)
    except NoAnswerException as e:
        print(e)
