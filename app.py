"""
Tweet Sentiment Classifier - Streamlit App
A simple web interface for classifying tweet sentiments using a fine-tuned transformer model.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page configuration
st.set_page_config(
    page_title="Tweet Sentiment Classifier",
    page_icon="🐦",
    layout="centered"
)

# Constants
# Replace this with your Hugging Face model repository name after uploading
# Example: "your-username/tweet-sentiment-classifier"
MODEL_NAME = os.getenv("MODEL_NAME", "ranashch/tweet-sentiment-classifier")
LOCAL_MODEL_PATH = "./models/tweet-sentiment-classifier"


@st.cache_resource
def load_model():
    """Load the trained model and tokenizer (cached for performance)."""
    try:
        # Try to load from local path first (for local development)
        if os.path.exists(LOCAL_MODEL_PATH):
            with st.spinner("Loading model from local directory..."):
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
                model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
                model.eval()
                st.success("✅ Model loaded successfully from local directory!")
                return tokenizer, model

        # Otherwise, load from Hugging Face Hub (for deployment)
        with st.spinner("Loading model from Hugging Face Hub... (this may take a moment on first load)"):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            model.eval()  # Set to evaluation mode

        st.success("✅ Model loaded successfully!")
        return tokenizer, model

    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.warning("""
        **This looks like a version compatibility issue!**

        Please run the fix script:
        1. Open Command Prompt / Terminal
        2. Run: `fix_windows.bat` (Windows)
        3. Or follow instructions in TROUBLESHOOTING.md
        """)
        with st.expander("🔧 Quick Fix Instructions"):
            st.code("""
# Run these commands in order:
pip uninstall numpy torch transformers -y
pip install numpy==1.24.3
pip install torch==2.1.0
pip install transformers==4.36.0
pip install streamlit==1.29.0
            """, language="bash")
        return None, None

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("See TROUBLESHOOTING.md for help")
        return None, None


def predict_sentiment(text, tokenizer, model):
    """
    Predict the sentiment of the input text.
    Returns: sentiment label and confidence score.
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()

    # Map prediction to label
    sentiment_map = {0: "Negative", 1: "Positive"}
    sentiment = sentiment_map[prediction]

    return sentiment, confidence


def main():
    """Main Streamlit app."""

    # Header
    st.title("🐦 Tweet Sentiment Classifier")
    st.markdown("""
    This app classifies the sentiment of tweets as **Positive** or **Negative** using a
    fine-tuned DistilBERT model trained on Twitter data.
    """)

    st.markdown("---")

    # Load model
    tokenizer, model = load_model()

    if tokenizer is None or model is None:
        st.stop()

    # Input section
    st.subheader("Enter a tweet to analyze:")

    # Text input
    tweet_text = st.text_area(
        "Tweet text",
        placeholder="Type or paste a tweet here...",
        height=100,
        label_visibility="collapsed"
    )

    # Example tweets
    st.markdown("**Or try an example:**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("😊 Positive Example"):
            tweet_text = "I absolutely love this! Best day ever! The weather is amazing and I'm feeling great!"

    with col2:
        if st.button("😞 Negative Example"):
            tweet_text = "This is terrible. Worst experience ever. So disappointed and frustrated."

    # Predict button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if tweet_text.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(tweet_text, tokenizer, model)

                # Display results
                st.markdown("---")
                st.subheader("Results:")

                # Create columns for better layout
                result_col1, result_col2 = st.columns([1, 1])

                with result_col1:
                    st.metric("Sentiment", sentiment)

                with result_col2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                # Visual indicator
                if sentiment == "Positive":
                    st.success("✅ This tweet has a **positive** sentiment!")
                    st.progress(confidence)
                else:
                    st.error("❌ This tweet has a **negative** sentiment!")
                    st.progress(confidence)

                # Show the analyzed tweet
                st.markdown("**Analyzed Tweet:**")
                st.info(tweet_text)

        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### About this Project
    - **Model:** DistilBERT fine-tuned on Twitter sentiment data
    - **Classes:** Negative (0) and Positive (1)
    - **Dataset:** Sentiment140 (Twitter sentiment dataset)

    **Note:** This is a binary sentiment classifier. Neutral sentiments may be classified
    as either positive or negative based on the model's interpretation.
    """)

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        ### How it works:
        1. Enter a tweet or text
        2. Click "Analyze Sentiment"
        3. Get instant sentiment prediction with confidence score

        ### Model Details:
        - Base model: DistilBERT
        - Fine-tuned on Twitter data
        - Binary classification (Positive/Negative)

        ### Tips:
        - Works best with short texts (tweet-length)
        - More expressive language = higher confidence
        - Emoji and slang are understood
        """)

        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit and HuggingFace Transformers")


if __name__ == "__main__":
    main()
