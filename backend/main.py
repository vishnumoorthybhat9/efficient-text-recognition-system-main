"""FastAPI application factory and entry point.

Run it with::

    python scripts/run_api.py
    # or
    uvicorn backend.main:app --reload

Start-up is deliberately cheap. The database schema is created, the ML
runtime discovers which models exist, and that is all -- **no model weights
are loaded and no training is started**. The first request for a model pays
for loading it, which keeps the server usable on a machine where nothing has
been trained yet.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.api.routes import datasets, evaluation, health, history, models, ocr, training
from backend.core.config import get_settings
from backend.core.errors import register_exception_handlers
from backend.db.session import dispose_engine, init_db
from backend.services.runtime import get_runtime, init_runtime, shutdown_runtime
from ml.utils.logging import setup_logging

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)

API_PREFIX = "/api"

DESCRIPTION = """
Backend for **Efficient Text Recognition from Images Using Deep Learning and
Self-Supervised Learning**.

### What this API exposes

* **Recognition** -- upload an image, get text, with per-line boxes, confidence
  and a full timing breakdown.
* **Preprocessing** -- every enhancement stage as an image, so the controls
  can be judged rather than guessed at.
* **Models** -- the project's CRNN+BiLSTM+Attention and Mamba recognisers
  alongside pretrained TrOCR. Every model carries a `provenance` field:
  `trained_in_project` or `pretrained_third_party`.
* **Evaluation** -- CER, WER, accuracy, latency, memory and parameter counts
  on a held-out split, plus the controlled SimCLR-vs-scratch comparison.
* **Training** -- start the three experiment stages; one job at a time.
* **History** -- every recognition, with optional ground truth for scoring.

### Honesty

Numbers come from running models. TrOCR's weights are third-party and its
results are labelled as an external baseline, never as a result of this work.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start up and shut down the application's shared resources."""
    settings = get_settings()
    setup_logging(settings.log_level, log_file=settings.log_path, json_format=settings.log_json)

    logger.info("=" * 70)
    logger.info("%s v%s (%s)", settings.app_name, settings.app_version, settings.environment)
    logger.info("=" * 70)

    started = time.perf_counter()
    init_db(settings)
    runtime = init_runtime(settings)

    logger.info(
        "Ready in %.2fs | device=%s | %d model(s) discovered | dataset %s",
        time.perf_counter() - started,
        runtime.device,
        len(runtime.registry),
        "ready" if runtime.dataset_ready() else "not prepared",
    )
    if not len(runtime.registry):
        logger.warning(
            "No models are available. Train one with scripts/train_crnn.py, or install "
            "`transformers` to use the pretrained TrOCR baseline."
        )

    try:
        yield
    finally:
        logger.info("Shutting down")
        shutdown_runtime()
        dispose_engine()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        """Tag every request with an id and record how long it took.

        The id is echoed in the ``X-Request-ID`` header and included in error
        responses, so a user-visible failure can be matched to its traceback in
        ``artifacts/logs/api.log``.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handler builds the response; just make sure the
            # timing and id are logged before it propagates.
            logger.exception(
                "Request %s %s failed after %.1f ms [request_id=%s]",
                request.method,
                request.url.path,
                (time.perf_counter() - started) * 1000,
                request_id,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"

        # Health polling every second would otherwise drown the log.
        if request.url.path not in ("/api/health", "/health"):
            logger.debug(
                "%s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code, elapsed_ms
            )
        return response

    register_exception_handlers(application)

    for router in (
        health.router,
        models.router,
        ocr.router,
        history.router,
        evaluation.router,
        datasets.router,
        training.router,
    ):
        application.include_router(router, prefix=API_PREFIX)

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send a bare visit to the interactive API documentation."""
        return RedirectResponse(url="/docs")

    return application


app = create_app()
