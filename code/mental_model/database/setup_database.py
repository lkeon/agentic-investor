from mental_model.database.base import Base
from mental_model.database.connection import engine

# Import the database models so SQLAlchemy registers the tables.
from mental_model.fragments import db_models  # noqa: F401


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    create_tables()
