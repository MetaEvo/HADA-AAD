from abc import ABC, abstractmethod
from agent.llm import OPENAI_MODEL
from utils.thread_logger import ThreadLoggerManager


class AgentSystem(ABC):
    def __init__(
        self,
        model=OPENAI_MODEL,
        chat_history_file='./outputs/chat_history.md',
    ):
        self.model = model

        # Initialize logger and store it in thread-local storage
        self.logger_manager = ThreadLoggerManager(log_file=chat_history_file)
        self.log = self.logger_manager.log

        # Clear the log file
        with open(chat_history_file, 'w') as f:
            f.write('')

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
