# 🐦 Tweet Sentiment Classifier

A machine learning project that classifies tweet sentiments as **Positive** or **Negative** using a fine-tuned DistilBERT transformer model. The project includes a user-friendly Streamlit web interface for real-time sentiment analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

This project demonstrates a practical application of Natural Language Processing (NLP) for sentiment analysis on Twitter data. It uses transfer learning with a pre-trained DistilBERT model, fine-tuned on the Sentiment140 dataset to classify tweets as positive or negative.

## ✨ Features

- **Pre-trained Transformer Model**: Uses DistilBERT for accurate sentiment classification
- **Interactive Web Interface**: Built with Streamlit for easy deployment and usage
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Sample Examples**: Pre-loaded positive and negative examples for testing
- **Clean Code Structure**: Well-organized and documented codebase

## 🖼️ Demo

The Streamlit app provides an intuitive interface where users can:
1. Enter any tweet or text
2. Click "Analyze Sentiment"
3. View the predicted sentiment (Positive/Negative) with confidence score
4. Try pre-loaded examples

## 🚀 Installation

### Prerequisites
- **Conda** (Anaconda or Miniconda) - **RECOMMENDED**
- OR Python 3.11 with pip

### Setup Method 1: Conda (Recommended - Cleanest!)

**Windows (double-click):**
```
setup_conda.bat
```

**Mac/Linux:**
```bash
bash setup_conda.sh
```

**Or manually:**
```bash
conda env create -f environment.yml
conda activate tweet-sentiment
```

**Why Conda?** No dependency conflicts, isolated environment, guaranteed compatible versions!

See `QUICKSTART.md` or `CONDA_SETUP.md` for detailed instructions.

---

### Setup Method 2: pip/venv (Alternative)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tweet-sentiment-classifier.git
   cd tweet-sentiment-classifier
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-fixed.txt
   ```

**Note:** If you encounter version conflicts, use the Conda method instead!

## 💻 Usage

**Important:** If using Conda, always activate the environment first:
```bash
conda activate tweet-sentiment
```

### Option 1: Use Pre-trained Model (Quickest - No Training!)

If you want to skip training and use a pre-trained sentiment model:

```bash
python use_pretrained.py
```

Then run the app:
```bash
streamlit run app.py
```

This downloads a DistilBERT model already fine-tuned on sentiment data. Perfect for testing!

### Option 2: Train Your Own Model (Recommended)

To train the sentiment classifier on Twitter data:

```bash
python train.py
```

**Note**: Training will:
- Automatically download the tweet_eval dataset (or fallback to emotion dataset)
- Use a sample of 10,000 tweets for faster training (configurable)
- Fine-tune DistilBERT for 2 epochs
- Save the model to `./models/tweet-sentiment-classifier/`
- Training takes ~10-15 minutes on CPU (faster with GPU)

### Option 3: Train from CSV (If Automatic Download Fails)

If you encounter dataset loading issues:

1. Download the Sentiment140 dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Place the CSV file at `./data/sentiment140.csv`
3. Run:
```bash
python train_from_csv.py
```

### Running the Streamlit App

After training (or downloading the pre-trained model), launch the web interface:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Troubleshooting

**Error: "Dataset scripts are no longer supported"**
- The updated `train.py` now uses supported datasets (tweet_eval or emotion)
- If issues persist, use Option 1 (pre-trained) or Option 3 (CSV)

**Error: "Model not found"**
- Make sure to run one of the training scripts first
- Or use `python use_pretrained.py` to download a pre-trained model

**Out of memory during training**
- Reduce `BATCH_SIZE` in train.py (line 26) from 16 to 8 or 4
- Reduce `sample_size` in train.py (line 64) to a smaller number

## 📁 Project Structure

```
tweet-sentiment-classifier/
│
├── app.py                          # Streamlit web application
├── train.py                        # Model training script (auto-download datasets)
├── train_from_csv.py               # Training script for manual CSV data
├── use_pretrained.py               # Download pre-trained model (no training)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
├── data/                           # Dataset directory
│   └── (auto-downloaded or place CSV here)
│
├── models/                         # Saved models directory
│   └── tweet-sentiment-classifier/ # Fine-tuned model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       └── training_summary.txt
│
└── notebooks/                      # Jupyter notebooks (optional)
    └── exploration.ipynb
```

## 🤖 Model Details

### Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Model Type**: Sequence Classification
- **Number of Labels**: 2 (Binary: Negative=0, Positive=1)
- **Max Sequence Length**: 128 tokens

### Training Configuration
- **Dataset**: tweet_eval sentiment (Twitter dataset) or emotion dataset (fallback)
- **Training Samples**: ~8,000 samples
- **Validation Samples**: ~2,000 samples
- **Epochs**: 2
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with weight decay

### Why DistilBERT?
- 40% smaller than BERT
- 60% faster than BERT
- Retains 97% of BERT's performance
- Perfect for deployment in resource-constrained environments

## 📊 Results

After training, you'll see metrics like:

```
Evaluation Results:
  accuracy: 0.8XXX
  f1: 0.8XXX
  precision: 0.8XXX
  recall: 0.8XXX
```

*Note: Actual results may vary based on the training sample and random seed.*

## 🌐 Deployment

### Deploying on Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

**Important**: Make sure to train and commit the model to the `models/` directory before deploying, or train it in the cloud environment.

### Alternative Deployment Options
- **Hugging Face Spaces**: Free hosting for ML apps
- **Heroku**: Cloud platform with free tier
- **AWS/GCP/Azure**: For production-scale deployments

## 🔮 Future Improvements

- [ ] Add support for neutral sentiment classification (3-class)
- [ ] Implement batch prediction for multiple tweets
- [ ] Add visualization of attention weights
- [ ] Include model comparison (different architectures)
- [ ] Add API endpoint for programmatic access
- [ ] Implement real-time Twitter stream analysis
- [ ] Add multilingual support
- [ ] Create Docker container for easy deployment
- [ ] Add more detailed analytics and statistics

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-trained models and training utilities
- **Streamlit**: Web app framework
- **Scikit-learn**: Evaluation metrics
- **Pandas & NumPy**: Data manipulation
- **Datasets (HuggingFace)**: Dataset loading and processing

## 📝 Notes

- The model is trained on English tweets and works best with English text
- Very short texts (< 5 words) may have lower confidence scores
- The model performs binary classification, so neutral sentiments will be classified as either positive or negative
- Training on the full dataset will improve accuracy but requires more time and computational resources

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for the Transformers library
- [Sentiment140](http://help.sentiment140.com/) for the dataset
- [Streamlit](https://streamlit.io/) for the amazing web framework

---

Made with ❤️ and Python
