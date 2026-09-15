from six.moves.urllib import response
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_summary(reviews_text,instruction="Write a short, honest summary (3-4 sentences) covering overall sentiment, key positives, key complaints, and any notable patterns."):

    prompt = f"""You are a business analyst summarizing customer feedback for a busy executive.

{instruction}

Reviews:
{reviews_text}

Summary:"""

    response=client.models.generate_content(
        model = "gemini-3.6-flash",
        contents = prompt
    )
    return response.text


def batch_summarize(all_reviews, batch_size=50):
    batche_summaries = []
    for i in range(0,len(all_reviews),batch_size):
        batch = all_reviews[i:i+batch_size]
        batch_text = "\n".join(batch)
        summary = generate_summary(batch_text)
        batche_summaries.append(summary)
    return batche_summaries


def final_summary(batch_summarize):
    combined = "\n\n".join(batch_summarize)
    instruction="You are given several partial summaries of different batches of customer reviews. Synthesize them into ONE final executive summary (4-5 sentences) covering overall sentiment, key positives, key complaints, and any notable patterns across the full dataset."
    return generate_summary(combined,instruction=instruction)
