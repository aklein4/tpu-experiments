

import logging

from utils import constants

class OnlyMain(logging.Filter):

    def __init__(self):
        super().__init__()

        self.is_main = constants.PROCESS_IS_MAIN()

    def filter(self, record: logging.LogRecord) -> bool:
        return self.is_main
    