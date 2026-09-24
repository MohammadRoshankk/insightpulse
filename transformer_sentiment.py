from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
def transformer_label(text):
    result = classifier(text)
    label = result[0]['label']
    return label.lower()

