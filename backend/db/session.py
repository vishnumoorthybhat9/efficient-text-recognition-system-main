"""Database engine and session management.

SQLite specifics that matter here
---------------------------------
* ``check_same_thread=False`` -- FastAPI runs sync endpoints in a thread pool,
  so a connection is created on one thread and used on another.
* **WAL journal mode** -- the default rollback journal takes a global write
  lock, which means a long-running training job writing progress would block
  every read.  WAL lets readers continue during a write.
* ``busy_timeout`` -- rather than failing instantly with "database is locked"
  when two writes collide, wait a few seconds.
* **Foreign keys are off by default in SQLite** and must be enabled per
  connection, otherwise ``ondelete="CASCADE"`` silently does nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import Settings, get_settings
from backend.db.models import Base

__all__ = ["init_engine", "get_engine", "get_sessionmaker", "session_scope", "init_db", "dispose_engine"]

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _configure_sqlite(engine: Engine) -> None:
    """Apply the per-connection PRAGMAs SQLite needs for this workload."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()


def init_engine(settings: Settings | None = None, *, echo: bool | None = None) -> Engine:
    """Create (or return) the global engine."""
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    config = settings or get_settings()
    url = config.resolved_database_url
    is_sqlite = url.startswith("sqlite")

    connect_args: dict[str, Any] = {"check_same_thread": False} if is_sqlite else {}

    _engine = create_engine(
        url,
        echo=bool(echo) if echo is not None else False,
        future=True,
        connect_args=connect_args,
        # SQLite has no server-side pool to exhaust; the default pool is fine
        # and `pool_pre_ping` catches a file that vanished underneath us.
        pool_pre_ping=True,
    )

    if is_sqlite and ":memory:" not in url:
        _configure_sqlite(_engine)

    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)

    logger.info("Database engine ready (%s)", url.split("///")[-1] if is_sqlite else url)
    return _engine


def get_engine() -> Engine:
    """The global engine, created on first call."""
    return _engine if _engine is not None else init_engine()


def get_sessionmaker() -> sessionmaker[Session]:
    """The global session factory."""
    if _session_factory is None:
        init_engine()
    assert _session_factory is not None
    return _session_factory


def init_db(settings: Settings | None = None) -> Engine:
    """Create the engine and every table that does not exist yet.

    ``create_all`` is used rather than a migration tool: the schema ships with
    the application, there is no deployed database to migrate, and adding
    Alembic would be ceremony with no payoff at this scale.  Changing a column
    type means deleting ``artifacts/ocr.db``, which the README states.
    """
    engine = init_engine(settings)
    Base.metadata.create_all(bind=engine)

    with engine.connect() as connection:
        tables = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).scalars().all() if engine.dialect.name == "sqlite" else []

    logger.info("Database initialised with %d table(s): %s", len(tables), ", ".join(tables))
    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for use outside a request.

    Commits on success, rolls back on any exception.

    >>> with session_scope() as session:
    ...     session.add(record)
    """
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Database transaction rolled back")
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Deliberately does *not* commit: endpoints commit explicitly, so a handler
    that raises after a partial write cannot leave it persisted.
    """
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def dispose_engine() -> None:
    """Close every pooled connection.  Used at shutdown and between tests."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        logger.debug("Database engine disposed")
    _engine = None
    _session_factory = None
