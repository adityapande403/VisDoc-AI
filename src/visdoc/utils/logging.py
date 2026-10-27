import logging
import sys
from rich.logging import RichHandler

def setup_logger(name: str = "visdoc", level: int = logging.INFO) -> logging.Logger:
    """Configures a structured, rich logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()
