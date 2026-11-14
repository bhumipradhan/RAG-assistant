from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base


def create_db_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kwargs: dict[str, object] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool  # one shared connection for in-memory DBs
        engine = create_engine(url, **kwargs)
    else:
        engine = create_engine(url, pool_pre_ping=True)
    # Fine for this project. In production use Alembic migrations instead.
    Base.metadata.create_all(engine)
    return engine


def create_session_factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=create_db_engine(url), expire_on_commit=False)
