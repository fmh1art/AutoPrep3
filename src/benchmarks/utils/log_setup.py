from __future__ import annotations

import logging
import os


def build_log_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


def configure_main_logger(tmp_root: str, log_filename: str = "main_log.ansi") -> str:
    os.makedirs(tmp_root, exist_ok=True)
    log_path = os.path.join(tmp_root, log_filename)

    root_logger = logging.getLogger()
    if any(
        getattr(handler, "_autoprep_log_path", None) == log_path
        for handler in root_logger.handlers
    ):
        return log_path

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(build_log_formatter())
    handler._autoprep_log_path = log_path  # type: ignore[attr-defined]
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    return log_path