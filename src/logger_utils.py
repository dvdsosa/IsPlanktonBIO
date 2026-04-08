import logging

ENABLE_PRINTING = False
ENABLE_PRINTING_YELLOW = False
ENABLE_PRINTING_RED = False
ENABLE_PLOT_DISPLAY = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_log_yellow(text: str) -> None:
    if ENABLE_PRINTING_YELLOW:
        logger.warning(text)


def print_log_red(text: str) -> None:
    if ENABLE_PRINTING_RED:
        logger.error(text)


def print_normal(text: str) -> None:
    if ENABLE_PRINTING:
        logger.info(text)


def print_green(text: str) -> None:
    if ENABLE_PRINTING:
        logger.info(text)


def print_yellow(text: str) -> None:
    if ENABLE_PRINTING_YELLOW:
        logger.warning(text)


def print_red(text: str) -> None:
    if ENABLE_PRINTING_RED:
        logger.error(text)
