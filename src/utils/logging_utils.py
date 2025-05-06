import logging
import os
import sys
import time
from datetime import datetime
import traceback
import json
import pprint as pp
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)


# Define custom exception to prevent multiple error logging
class LoggedException(Exception):
    """Custom exception to prevent multiple error logging."""

    def __init__(self):
        super().__init__()


# Define color mapping for different log levels
LOG_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT,
}


# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages in console output"""

    def format(self, record):
        levelno = record.levelno
        if levelno in LOG_COLORS:
            record.levelname = (
                f"{LOG_COLORS[levelno]}{record.levelname}{Style.RESET_ALL}"
            )
            record.msg = f"{LOG_COLORS[levelno]}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


# Configure logger
def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Configure and return a logger with consistent formatting

    Args:
        name: Logger name, typically __name__ of the calling module
        log_level: Logging level (default: INFO)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist and no specific path is provided
    if log_file is None:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join("logs", f"{timestamp}.log")

    # Create formatter with detailed information
    file_formatter = logging.Formatter(
        "%(asctime)s - [%(name)s:%(lineno)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create colored formatter for console output
    console_formatter = ColoredFormatter(
        "%(asctime)s - [%(name)s:%(lineno)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    # Configure console handler with colored format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates when reconfiguring
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_variable(variable_name, variable_value, additional_info=None):
    """
    Log a variable value to a dedicated variables log file with execution timestamp

    Args:
        variable_name: Name of the variable to log
        variable_value: Value of the variable to log
        additional_info: Optional additional context information

    Returns:
        Path to the variables log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Generate execution timestamp if not already created
    if not hasattr(log_variable, "execution_timestamp"):
        log_variable.execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create variables log file with execution timestamp
    var_log_file = os.path.join(
        "logs", f"variables_{log_variable.execution_timestamp}.log"
    )

    # Format the variable value for logging
    if isinstance(variable_value, (dict, list, tuple, set)):
        formatted_value = pp.pformat(variable_value, indent=2)
    else:
        formatted_value = str(variable_value)

    # Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {variable_name} = {formatted_value}"

    # Add additional info if provided
    if additional_info:
        if isinstance(additional_info, dict):
            additional_info_str = pp.pformat(additional_info, indent=2)
        else:
            additional_info_str = str(additional_info)
        log_entry += f"\nContext: {additional_info_str}"

    # Append to log file
    with open(var_log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n\n" + "-" * 80 + "\n\n")

    return var_log_file


def log_exception(logger, e, context=None):
    """
    Log detailed exception information

    Args:
        logger: Logger instance
        e: Exception object
        context: Optional dict with additional context about system state
    """
    exc_info = sys.exc_info()
    stack_trace = "".join(traceback.format_exception(*exc_info))

    # Create error details
    error_details = {
        "exception_type": type(e).__name__,
        "exception_message": str(e),
        "traceback": pp.pformat(stack_trace, indent=4),
    }

    # Add context if provided
    if context:
        error_details["context"] = context

    # Log the error with details
    logger.error(f"Exception occurred: {type(e).__name__}: {str(e)}")
    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")


def log_function_entry_exit(logger, level=logging.DEBUG):
    """
    Decorator for logging function entry and exit with timing information

    Args:
        logger: Logger instance
        level: Logging level for the entry/exit logs

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Entering {func_name}")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(level, f"Exiting {func_name} (took {elapsed:.3f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.log(level, f"Exception in {func_name} after {elapsed:.3f}s")
                log_exception(logger, e)
                raise

        return wrapper

    return decorator
