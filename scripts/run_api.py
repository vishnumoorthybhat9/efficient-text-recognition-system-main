"""Start the FastAPI server.

A thin wrapper around uvicorn that reads host/port/reload from ``.env`` (via
:class:`backend.core.config.Settings`) so there is one place to configure the
server, and so it can be started from anywhere without setting PYTHONPATH.

::

    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --reload
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_settings  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("run_api")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run the OCR API server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.reload,
        help="restart on source changes (development only)",
    )
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    parser.add_argument("--log-level", default=settings.log_level.lower())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level.upper())

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed. Install it with:  pip install 'uvicorn[standard]'")
        return 1

    settings = get_settings()
    logger.info("Starting %s on http://%s:%d", settings.app_name, args.host, args.port)
    logger.info("Interactive API docs: http://%s:%d/docs", args.host, args.port)

    # `reload` needs the import string, not the app object, so the reloader can
    # re-import the module in the child process.
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
        # Our own logging config is already installed; letting uvicorn install
        # its own would duplicate every line.
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
