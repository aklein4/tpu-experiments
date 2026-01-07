
import sys
import inspect

from base_handler import BaseHandler


def format_chat(data):
    roles = [msg["role"] for msg in data]
    content = [msg["content"].strip() for msg in data]

    if roles[:2] == ["user", "assistant"]:
        return f"User:\n\n{content[0]}\n\nAssistant:\n\n", content[1]

    elif roles[:3] == ["system", "user", "assistant"]:
        return f"System:\n\n{content[0]}\n\nUser:\n\n{content[1]}\n\nAssistant:\n\n", content[2]

    else:
        return None, None



""" ===== Chat ===== """

class SmolTalkHandler(BaseHandler):

    url = "HuggingFaceTB/smoltalk"
    subset = "all"
    split = "train"


    def map(self, example):
        return format_chat(example["data"]) + ("chat",)



HANDLERS = [x[1] for x in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(x[1]) and issubclass(x[1], BaseHandler) and x[1] is not BaseHandler]
