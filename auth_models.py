from database import Base, engine, SessionLocal
from sqlalchemy import Column ,Integer , String ,Boolean , DateTime
from datetime import datetime

class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer , primary_key=True)
    name = Column(String)
    api_key = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow) 