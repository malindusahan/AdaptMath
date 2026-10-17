import logging


_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)


def configure_logging(
    level: str = "INFO",
) -> None:
    """
    Configure application logging once using Python's
    standard logging package.

    Request bodies, learner answers, prompts, expected
    answers, and learner-memory payloads must not be logged.
    """

    resolved_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )

    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format=_LOG_FORMAT,
        )
    else:
        root_logger.setLevel(
            resolved_level
        )