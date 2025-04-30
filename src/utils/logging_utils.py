import logging
import os
import sys
import time
from datetime import datetime
import traceback
import json

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
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s:%(lineno)d] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Configure console handler with the same format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
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

def log_exception(logger, e, context=None):
    """
    Log detailed exception information
    
    Args:
        logger: Logger instance
        e: Exception object
        context: Optional dict with additional context about system state
    """
    exc_info = sys.exc_info()
    stack_trace = ''.join(traceback.format_exception(*exc_info))
    
    # Create error details
    error_details = {
        "exception_type": type(e).__name__,
        "exception_message": str(e),
        "traceback": stack_trace
    }
    
    # Add context if provided
    if context:
        error_details["context"] = context
    
    # Log the error with details
    logger.error(f"Exception occurred: {type(e).__name__}: {str(e)}")
    logger.debug(f"Error details: {json.dumps(error_details, indent=2)}")

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