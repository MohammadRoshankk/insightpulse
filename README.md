# InsightPulse

InsightPulse is an AI-powered customer review intelligence platform that helps businesses understand what their customers are actually saying — automatically analyzing sentiment, surfacing key topics, detecting complaints, and tracking trends over time, all through an interactive dashboard.

This project was built as a hands-on learning journey through data science and NLP fundamentals — starting from raw Python file handling and progressing through Pandas, exploratory data analysis, sentiment analysis, keyword extraction, and machine learning tooling — with every feature grounded in a real business use case rather than isolated tutorial exercises.

## Features

- **Data Cleaning** — handles missing values, duplicate detection, and text normalization on raw review exports
- **Exploratory Data Analysis** — rating distributions and review-length patterns via Pandas and Matplotlib
- **Sentiment Analysis** — a hand-built rule-based baseline (41% accuracy) upgraded to NLTK's VADER (63.7% accuracy), with documented, evidence-based evaluation of each approach's strengths and limitations
- **Keyword & Topic Extraction** — word frequency analysis, bigram extraction, and TF-IDF vectorization via scikit-learn
- **Complaint & Feature Request Detection** — rule-based classification to flag actionable customer feedback
- **Trend Analysis** — time-series analysis of rating and review volume trends using Pandas datetime operations
- **Interactive Dashboard** — a Streamlit web app consolidating all analyses into a single, non-technical-user-facing interface

## Tech Stack

- **Python** — core language
- **Pandas / NumPy** — data manipulation and analysis
- **Matplotlib** — data visualization
- **NLTK (VADER)** — sentiment analysis
- **scikit-learn** — TF-IDF vectorization
- **Streamlit** — interactive dashboard
- **Git/GitHub** — version control

## Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/MohammadRoshankk/insightpulse.git
cd insightpulse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the analysis pipeline:

```bash
python3 analyse_sentiment.py
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

## Project Structure

- `data/`: Contains sample reviews and datasets
- `analyse_sentiment.py`: Main analysis script
- `dashboard.py`: Streamlit application
- `clean_text.py`: Utilities for text processing
- `vader_sentiment.py`: Sentiment analysis implementation
- `detect_category.py`: Rule-based complaint/feature request detection

## What I Learned

This project was built incrementally, milestone by milestone, with an emphasis on understanding *why* each technique works (and where it fails) rather than just implementing it. Along the way, this included:

- Debugging real bugs (variable scope errors, early-return logic bugs, incorrect exception handling)
- Learning to sanity-check results rather than trusting output blindly — including catching a false "75% duplicate rate" caused by a flawed detection strategy, and a misleading "100% accuracy" result caused by a fallback-default bias
- Making and testing a data-driven hypothesis (widening VADER's neutral threshold) that turned out to be *wrong* — and learning why negative results are still valuable
- Building good engineering habits: version control, `.gitignore` hygiene, virtual environments, and separating reusable code from experimentation

## Screenshots

*(Add a screenshot of your dashboard here — drag an image into this README.md file on GitHub's web editor, or use `![Dashboard](path/to/screenshot.png)`)*
