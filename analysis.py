from my_model import load_ai_models
from newspaper import Article

def get_article_text(url):
    """
    Extracts text from a URL using newspaper.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text, article.authors, article.publish_date
    except Exception as e:
        return None, None, None, None

def run_ai_analysis(article_text):
    """
    Orchestrates the AI models to process the text.
    """
    # 1. Get the cached models
    # 'summarizer_func' is the custom function
    # 'sentiment_pipe' is the standard Hugging Face pipeline
    summarizer_func, sentiment_pipe = load_ai_models()
    
    # 2. Generate Summary
    # Passed the custom function, so it just call it like a normal Python function
    summary = summarizer_func(article_text)
    
    # 3. Analyze Sentiment
    # Truncate to 512 tokens because BERT models crash on long text
    # The pipeline returns a list like [{'label': 'POSITIVE', 'score': 0.99}]
    sentiment_result = sentiment_pipe(article_text[:512])[0]
    
    return {
        "summary": summary,
        "sentiment_label": sentiment_result['label'],
        "sentiment_score": sentiment_result['score']
    }