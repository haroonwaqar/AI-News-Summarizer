import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

@st.cache_resource
def load_ai_models():
    # 1. Setup Device (Apple Silicon Mac acceleration)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # PART A: Summarization (Manual Implementation)
    # NOT using pipeline("summarization") here because it crashes on setup.
    summ_model_name = "sshleifer/distilbart-cnn-12-6"
    print("Loading Summarization Model directly...")
    
    # Load the components manually
    tokenizer = AutoTokenizer.from_pretrained(summ_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(summ_model_name).to(device)

    # Custom function to replace the pipeline
    def custom_summarizer(text):
        clean_text = text.strip().replace("\n", " ")
        inputs = tokenizer(clean_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


    # "sentiment-analysis" supported, so use the standard pipeline here.
    print("Loading Sentiment Model...")
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    return custom_summarizer, sentiment_pipe