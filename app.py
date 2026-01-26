import streamlit as st
from analysis import get_article_text, run_ai_analysis

st.set_page_config(page_title="AI News Summarizer", page_icon="🤖")

st.title("🤖 AI News Summarizer")
st.markdown("Powered by **DistilBART** (Summarization) & **DistilBERT** (Sentiment)")

# 1. Input Section
url = st.text_input("Paste News Article URL:", placeholder="https://www.dawn.com/...")

# 2. Button Logic
if st.button("Analyze Article"):
    if not url:
        st.warning("Please enter a URL first.")
    else:
        with st.spinner("🔍 Reading & Analyzing... "):
            try:
                # A. Extract Text
                title, text, authors, date = get_article_text(url)
                
                if not text:
                    st.error("Could not extract text. The website might block bots.")
                else:
                    # B. Run AI Analysis
                    results = run_ai_analysis(text)
                    
                    # C. Display Results
                    st.success("Analysis Complete!")
                    st.subheader(title)
                    
                    # Metadata row
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Author:** {', '.join(authors) if authors else 'Unknown'}")
                    with col2:
                        st.write(f"**Date:** {date if date else 'Unknown'}")
                    
                    st.divider()
                    
                    # Summary Section
                    st.markdown("### AI Summary")
                    st.write(results["summary"])
                    
                    st.divider()
                    
                    # Sentiment Section
                    st.markdown("### Market Sentiment")
                    # Color-code the sentiment
                    sentiment_color = "green" if results["sentiment_label"] == "POSITIVE" else "red"
                    st.markdown(f"**Verdict:** :{sentiment_color}[{results['sentiment_label']}]")
                    st.progress(results["sentiment_score"], text=f"Confidence: {results['sentiment_score']:.2f}")

            except Exception as e:
                st.error(f"An error occurred: {e}")