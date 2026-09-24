from database import Review,SessionLocal
#from vader_sentiment import vader_label
from detect_category import detect_category
from clean_text import clean_text
from transformer_sentiment import transformer_label



session = SessionLocal()

reviews = session.query(Review).all()

for review in reviews:
    review.sentiment = transformer_label(review.text)
    review.category = detect_category(clean_text(review.text))

session.commit()
session.close()


print("Updated sentiment and category for all reviews")
