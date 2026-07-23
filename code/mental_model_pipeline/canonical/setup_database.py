"""Create or recreate the PostgreSQL tables required by the canonical-model pipeline."""

from __future__ import annotations

import argparse

from sqlalchemy import text

from mental_model_pipeline.canonical import db_models as canonical_models  # noqa: F401
from mental_model_pipeline.database.base import Base
from mental_model_pipeline.database.connection import engine
from mental_model_pipeline.fragments import db_models as fragment_models  # noqa: F401


LEGACY_AND_CURRENT_TABLES = (
    "canonical_model_edges",
    "canonical_model_fragments",
    "canonical_mental_models",
    "canonical_pipeline_runs",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the canonical-model MVP database tables."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Drop current and legacy canonical tables before creating the "
            "MVP schema. This deletes canonical data."
        ),
    )
    return parser.parse_args()


def create_tables(*, recreate: bool) -> None:
    if recreate:
        with engine.begin() as connection:
            for table_name in LEGACY_AND_CURRENT_TABLES:
                connection.execute(
                    text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
                )

    Base.metadata.create_all(bind=engine)
    print("Canonical MVP database tables created successfully.")


if __name__ == "__main__":
    args = parse_arguments()
    create_tables(recreate=args.recreate)
