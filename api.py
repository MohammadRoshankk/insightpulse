from database import SessionLocal
from auth_models import Client
from pydantic import BaseModel
from transformer_sentiment import transformer_label

from detect_category import detect_category
from clean_text import clean_text
import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException, FastAPI, Depends

from database import Base, engine
import auth_models  # registers the Client table with Base

Base.metadata.create_all(engine)  # creates any missing tables at startup


load_dotenv()

def verify_api_key(x_api_key: str = Header(...)):
    session = SessionLocal()
    client = session.query(Client).filter(Client.api_key == x_api_key, Client.is_active == True).first()
    session.close()

    if not client:
        raise HTTPException(status_code=401, detail="Invalid API Key")






app = FastAPI()

@app.get("/")

def read_root():
    return {"message": "InsightPulse API is running"}

class ReviewInput(BaseModel):
    text:str

@app.post("/analyze")
def analyze_review(review: ReviewInput, key: str = Depends(verify_api_key)):
    cleaned=clean_text(review.text)
    sentiment = transformer_label(review.text)
    category = detect_category(cleaned)

    return {
        
        "text":review.text,
        "sentiment":sentiment,
        "category": category
        
        }