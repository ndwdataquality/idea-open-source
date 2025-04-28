import os

from singleton_decorator import singleton


@singleton
class LoggerProperties:
    log_level: str = os.environ.get("LOG_LEVEL") if os.environ.get("LOG_LEVEL") else "INFO"
