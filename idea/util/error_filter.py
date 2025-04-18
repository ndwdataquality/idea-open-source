import logging


class ErrorFilter:
    """Filters logging messages with level WARNING and lower."""

    @staticmethod
    def filter(record):
        return record.levelno < logging.WARNING
