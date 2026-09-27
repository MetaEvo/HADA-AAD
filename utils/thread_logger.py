import logging
from logging.handlers import RotatingFileHandler
import threading
import os

class ThreadLoggerManager:
    _loggers = {}
    _lock = threading.Lock()

    def __init__(self, log_file='./chat_history.md', level=logging.INFO):
        self.log_file = log_file
        self.level = level
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def get_logger(self):
        """
        Get or create a logger specific to (thread_id, log_file).
        """
        thread_id = threading.get_ident()
        key = (thread_id, self.log_file)

        with self._lock:
            if key not in self._loggers:
                logger = logging.getLogger(f'AgentSystem-{thread_id}-{self.log_file}')
                logger.setLevel(self.level)
                if not logger.handlers:
                    file_handler = RotatingFileHandler(
                        self.log_file, maxBytes=10 * 1024 * 1024, backupCount=5
                    )
                    file_handler.setFormatter(logging.Formatter('%(message)s'))
                    file_handler.setLevel(self.level)
                    logger.addHandler(file_handler)
                self._loggers[key] = logger

        return self._loggers[key]

    def log(self, message, level=logging.INFO):
        logger = self.get_logger()
        logger.log(level, message)
