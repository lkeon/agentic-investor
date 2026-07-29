import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required. Set it in .env locally or provide it "
        "through the deployment environment."
    )

engine = create_engine(
    DATABASE_URL,
    echo=False,
    # Hosted PostgreSQL services can close idle pooled connections. Validate a
    # connection before reuse and recycle it periodically so Streamlit reruns
    # and long committee executions do not inherit a stale socket.
    pool_pre_ping=True,
    pool_recycle=300,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)
