# AI-Powered News Summarizer

A news intelligence platform that leverages **Deep Learning** to read, understand, and synthesize global news in real-time.

Unlike traditional summarizers that simply extract sentences, this application uses **Abstractive Transformer Models** to generate human-like summaries and performs high-confidence **Sentiment Analysis** to gauge market/geopolitical risk.

## Key Features

* **Abstractive Summarization:** Powered by **Facebook's DistilBART** (`distilbart-cnn-12-6`), capable of rewriting and synthesizing complex articles into concise, readable briefs.
* **Contextual Sentiment Engine:** Uses **DistilBERT** to analyze the emotional tone (Positive/Negative) of news, providing a confidence score useful for market sentiment analysis.
* **Robust NLP Pipeline:** Features custom tokenization and chunking strategies to handle long-form journalism without crashing.

## Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **AI/ML Frameworks:** PyTorch (MPS Support), Hugging Face Transformers
* **Models:**
    * *Summarization:* `sshleifer/distilbart-cnn-12-6`
    * *Sentiment:* `distilbert-base-uncased-finetuned-sst-2-english`
* **Data Ingestion:** Newspaper (Web Scraping & Parsing)

## Project Structure

```text
├── app.py           # Frontend (UI): Handles user input and displays results
├── analysis.py      # Controller: Orchestrates data flow between scraper and AI
├── my_model.py      # AI Engine: Manages model loading, caching, and hardware acceleration
└── requirements.txt # Dependencies
```
## Screenshot

![Home](<screenshot/image.png>)
Link to the article: <https://www.dawn.com/news/196954626-countries-including-pakistan-named-founding-members-of-trump-led-board-of-peace>
