import logging
import sys
import time

import numpy as np
import pandas as pd

from idea.util.error_filter import ErrorFilter
from idea.util.logger_properties import LoggerProperties

logger = logging.getLogger(__name__)


class DatadogFormatter(logging.Formatter):
    """
    In order for Datadog to display a multiline log message correctly, the message needs to be
    escaped.
    """

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).encode("unicode_escape").decode("utf-8")


def configure_logger():
    properties = LoggerProperties()

    logging_format = "%(asctime)s %(levelname)s %(message)s"
    root_logger = logging.getLogger("")

    # Default loggers (typically for when the script runs locally)
    formatter = logging.Formatter(logging_format)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(formatter)
    err_handler.setLevel("WARNING")
    root_logger.addHandler(err_handler)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(formatter)
    out_handler.setLevel("DEBUG")
    out_handler.addFilter(ErrorFilter)
    root_logger.addHandler(out_handler)

    root_logger.setLevel("INFO")
    root_logger.info(f"Log level set to {properties.log_level}")
    root_logger.setLevel(properties.log_level)

    # Catch uncaught tracebacks, convert to critical logging.
    sys.excepthook = handle_exception

    # Ignore pandas chained assignment warnings
    pd.options.mode.chained_assignment = None


def ranges_to_datetime64(ranges):
    a = np.array(ranges)
    # Pandas can only convert 1d arrays to datetime, so the array is flattened and un-flattened
    # before and after the conversion
    # Pandas to_datetime conversion is much more flexible than numpy datetime64
    return pd.to_datetime(a.reshape(-1)).values.reshape(-1, 2)


def create_message_df(message: str) -> pd.DataFrame:
    return pd.DataFrame([[message]], columns=[""])


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))


def log_time(logging_level: str):
    # The outmost wrapper function, gets the logging level as string.
    # Returns the actual decorator function 'log(func)'.
    def log(func):
        # Real decorator function to log.
        def wrapped(*args, **kwargs):
            # The wrapped function that replaces the original function.

            # Validate the logging level.
            if logging_level.lower() not in ["debug", "info", "warning", "error", "critical"]:
                raise ValueError(
                    "Invalid logging level given. Must be one of "
                    '["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].'
                )

            # calculate the time passed while the main function is called
            start_time = time.time_ns()
            result = func(*args, **kwargs)
            end_time = time.time_ns()
            calculated_time = end_time - start_time

            # Get the logger correspond to the specified level and log the result.
            log_func = getattr(logger, logging_level.lower())
            log_func(f"Execution of {func.__name__} took {calculated_time} seconds.")

            return result

        return wrapped

    return log


def flatten(nested_list: list):
    """Flatten nested lists."""
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


def reverse_dictionary(mapping: dict) -> dict:
    """Reverse dictionary with single key, value items."""
    return {value: key for key, value in mapping.items()}
