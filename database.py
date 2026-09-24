
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///insightpulse.db")
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
