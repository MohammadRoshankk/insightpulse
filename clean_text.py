import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]","",text)
    return text


#print(clean_text("Item #2 didn't arrive on time... 3/5 stars!"))