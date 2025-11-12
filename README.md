# 🐦 Tweet Sentiment Classifier

A ready-to-deploy sentiment analysis application that classifies tweet sentiments as **Positive** or **Negative** using a fine-tuned DistilBERT transformer model. Features a user-friendly Streamlit web interface for real-time sentiment analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Usage Notes](#usage-notes)
- [Acknowledgements](#acknowledgements)

## 🎯 Overview

This project provides a production-ready sentiment analysis application for Twitter data. It uses a fine-tuned DistilBERT model hosted on Hugging Face Hub and can be deployed instantly to Streamlit Cloud or run locally for development.

## ✨ Features

- **Pre-trained Transformer Model**: Uses DistilBERT fine-tuned on Sentiment140 dataset
- **Interactive Web Interface**: Beautiful Streamlit UI for easy interaction
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Dual Loading Strategy**: Loads from local model or Hugging Face Hub automatically
- **Sample Examples**: Pre-loaded positive and negative tweet examples
- **Production Ready**: Optimized for deployment with error handling and caching

## 🖼️ Demo

The Streamlit app provides an intuitive interface where users can:
1. Enter any tweet or text in the input area
2. Click "Analyze Sentiment" or try example tweets
3. View the predicted sentiment (Positive/Negative) with confidence score
4. See visual indicators and progress bars for results

## 🚀 Quick Start

Get up and running in 2 minutes:

```bash
# Clone the repository
git clone https://github.com/yourusername/tweet-sentiment-classifier.git
cd tweet-sentiment-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will automatically download the model from Hugging Face Hub on first run!

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tweet-sentiment-classifier.git
   cd tweet-sentiment-classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Running the Application

Simply run:
```bash
streamlit run app.py
```

The app will:
1. Check for a local model in `./models/tweet-sentiment-classifier/`
2. If not found, automatically download from Hugging Face Hub (ranashch/tweet-sentiment-classifier)

### Using Your Own Model

If you have trained your own model:

1. Place your model files in `./models/tweet-sentiment-classifier/`
2. Ensure it includes: `config.json`, `pytorch_model.bin` or `.safetensors`, and tokenizer files
3. Run the app - it will use your local model automatically

## 📁 Project Structure

```
tweet-sentiment-classifier/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
└── models/                         # Saved models directory
    └── tweet-sentiment-classifier/ # Fine-tuned model files
        ├── config.json             # Model configuration
        ├── tokenizer.json          # Tokenizer data
        ├── tokenizer_config.json   # Tokenizer configuration
        ├── special_tokens_map.json # Special tokens mapping
        └── vocab.txt               # Vocabulary file
```

**Key Files:**
- `app.py` - Main Streamlit application with sentiment analysis UI

## 🤖 Model Details

### Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Model Type**: Sequence Classification
- **Number of Labels**: 2 (Binary: Negative=0, Positive=1)
- **Max Sequence Length**: 128 tokens
- **Model Size**: ~256 MB
- **Hugging Face Hub**: [ranashch/tweet-sentiment-classifier](https://huggingface.co/ranashch/tweet-sentiment-classifier)

### Training Details
- **Dataset**: Sentiment140 (Twitter sentiment dataset with 1.6M tweets)
- **Fine-tuning**: Trained on labeled positive/negative tweets
- **Preprocessing**: Tokenized using DistilBERT tokenizer
- **Framework**: PyTorch with Hugging Face Transformers

### Why DistilBERT?
- **40% smaller** than BERT-base
- **60% faster** inference time
- **Retains 97%** of BERT's language understanding
- **Low memory footprint** - perfect for web deployment
- **Production ready** - optimized for real-world applications

## 🌐 Deployment

This app is already deployed in Streamlit Cloud.
Here is the link: https://tweet-sentiment-classifier.streamlit.app/

### Quick Deploy to Streamlit Cloud

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy tweet sentiment classifier"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `tweet-sentiment-classifier`
   - Branch: `main`
   - Main file: `app.py`
   - Click "Deploy!"

3. **That's it!** The app will:
   - Install dependencies from `requirements.txt`
   - Download the model from Hugging Face Hub automatically
   - Be live at your custom Streamlit URL

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **PyTorch 2.0+** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained models and tokenizers
- **Streamlit** - Interactive web application framework
- **Hugging Face Hub** - Model hosting and distribution
- **NumPy & Pandas** - Data manipulation (for preprocessing)
- **Scikit-learn** - Evaluation metrics

## 📝 Usage Notes

- **Language**: Works best with English text (trained on English tweets)
- **Input Length**: Optimized for tweet-length text (up to 280 characters)
- **Binary Classification**: Classifies as Positive or Negative only (no neutral class)
- **Confidence Scores**: Higher confidence for more expressive language
- **Performance**: Very short texts (< 5 words) may have lower confidence
- **Special Characters**: Handles emojis, hashtags, and @mentions

## 🙏 Acknowledgments

- **Hugging Face** - For the amazing Transformers library and model hosting
- **Sentiment140** - For the comprehensive Twitter sentiment dataset
- **Streamlit** - For making web app development incredibly simple
- **DistilBERT Team** - For the efficient transformer model
- **Open Source Community** - For continuous support and contributions

---

**Made with Python and Streamlit**
