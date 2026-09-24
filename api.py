from fastapi import FastAPI
from pydantic import BaseModel
from transformer_sentiment import transformer_label

#from vader_sentiment import vader_label
from detect_category import detect_category
from clean_text import clean_text



app = FastAPI()

@app.get("/")

def read_root():
    return {"message": "InsightPulse API is running"}

class ReviewInput(BaseModel):
    text:str

@app.post("/analyze")
def analyze_review(review: ReviewInput):
    cleaned=clean_text(review.text)
    sentiment = transformer_label(review.text)
    category = detect_category(cleaned)

    return {
        
        "text":review.text,
        "sentiment":sentiment,
        "category": category
        
        }