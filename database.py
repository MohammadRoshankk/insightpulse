
import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

def get_database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    if os.getenv("CLOUD_MODE") == "1":
        if os.path.exists("data/public_reviews.db"):
            return "sqlite:///data/public_reviews.db"
    if not os.path.exists("insightpulse.db") and os.path.exists("data/public_reviews.db"):
        return "sqlite:///data/public_reviews.db"
    if os.path.exists("insightpulse.db") and os.path.exists("data/public_reviews.db"):
        try:
            import sqlite3
            conn = sqlite3.connect("insightpulse.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reviews")
            count = cursor.fetchone()[0]
            conn.close()
            if count == 0:
                return "sqlite:///data/public_reviews.db"
        except Exception:
            return "sqlite:///data/public_reviews.db"
    return "sqlite:///insightpulse.db"

engine = create_engine(get_database_url())
Base = declarative_base()

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    rating = Column(Float)
    date = Column(DateTime)
    sentiment = Column(String)
    category = Column(String)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
