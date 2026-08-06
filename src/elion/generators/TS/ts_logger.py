import logging
import sys
from typing import Optional

_FMT     = "%(asctime)s,%(msecs)d %(levelname)-8s %(pathname)s:%(lineno)d %(message)s"
_DATEFMT = "%Y-%m-%d:%H:%M:%S"


def get_logger(name: str = None, level: str = "INFO", filename: Optional[str] = None) -> logging.Logger:
    """
    Basic logger for this repo.
    :param name: usually the file name which can be passed to the get_logger function like this get_logger(__name__)
    :param level: logging level
    :param filename: Filename to write logging to. If None, logging will print to STDOUT.
    :return: logger

    STDOUT, not stderr — and that word is the whole point of this function.

    `logging.basicConfig()` with no `stream=` installs a StreamHandler on
    **sys.stderr**. Every '[warm_up] …' and '[evaluate] …' line this logger
    emits therefore went to stderr, while the Flask UI's live parser
    (ts_routes._drain_stdout -> _update_reagent -> _recompute_top5) reads
    **stdout**. The run's own debug summary said so plainly and nobody read it:

        warm_up_lines(stdout=0, stderr=47121)

    Consequence: zero reagents were parsed live, history["top5"] stayed empty,
    the session file was written with no rankings, and the TS Belief panel
    rendered "No reagent rankings yet" after a run that had just scored 47,619
    molecules perfectly well. The end-of-run checkpoint save was unaffected
    because it parses `_all_log_lines + _stderr_lines` — which is exactly why
    the checkpoint JSON was fine while the live panel was empty.

    Implementation note: this attaches an explicit handler to the NAMED logger
    and sets propagate=False, rather than passing stream= to basicConfig().
    basicConfig is a no-op once the root logger has handlers, and `force=True`
    would tear down whatever else configured logging first. This way the change
    is scoped to this logger and cannot stomp another subsystem's config.
    """
    if name is None:
        name = "TSLogger"

    if filename:
        # File sink requested — keep the original behaviour exactly.
        logging.basicConfig(format=_FMT, datefmt=_DATEFMT, filename=filename)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Idempotent: get_logger() is called per-module, and duplicate handlers
    # would emit every line N times into the parser.
    if not any(getattr(h, "_ts_stdout_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        handler._ts_stdout_handler = True
        logger.addHandler(handler)
    # Do not also bubble to the root logger's stderr handler — that would put
    # every line on BOTH streams and double-count them in the UI.
    logger.propagate = False
    return logger