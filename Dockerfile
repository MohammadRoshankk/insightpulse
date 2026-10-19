FROM python:3.14-slim

WORKDIR /app

# Install the CPU-only PyTorch build first (much smaller than the CUDA default)
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Install the remaining pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords so the NLP code works inside the container
RUN python -m nltk.downloader stopwords

# Download the DistilBERT model into the image, so the container doesn't need
# internet access at startup
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')"

# Copy the project code (.dockerignore controls what's excluded)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]