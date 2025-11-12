# Streamlit Deployment Guide

Complete step-by-step guide to deploy your Tweet Sentiment Classifier to Streamlit Cloud.

---

## Step 1: Upload Model to Hugging Face Hub

### 1.1 Create a Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Verify your email

### 1.2 Create Access Token
1. Go to Settings → Access Tokens
2. Click "New token"
3. Name it (e.g., "model-upload")
4. Select "Write" permission
5. Copy the token (save it securely)

### 1.3 Install Hugging Face CLI
```bash
pip install huggingface_hub
```

### 1.4 Login to Hugging Face
```bash
huggingface-cli login
```
Paste your access token when prompted.

### 1.5 Upload Your Model
Run this Python script to upload your model:

```python
from huggingface_hub import HfApi, create_repo

# Configuration
repo_name = "your-username/tweet-sentiment-classifier"  # Replace with your username
local_model_path = "./models/tweet-sentiment-classifier"

# Create repository
api = HfApi()
try:
    create_repo(repo_id=repo_name, repo_type="model", private=False)
    print(f"✅ Repository created: {repo_name}")
except Exception as e:
    print(f"Repository might already exist: {e}")

# Upload all files
api.upload_folder(
    folder_path=local_model_path,
    repo_id=repo_name,
    repo_type="model"
)

print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_name}")
```

**OR** use the CLI:
```bash
huggingface-cli upload your-username/tweet-sentiment-classifier ./models/tweet-sentiment-classifier
```

### 1.6 Update app.py
After uploading, edit `app.py` line 21:
```python
MODEL_NAME = os.getenv("MODEL_NAME", "your-username/tweet-sentiment-classifier")
```

---

## Step 2: Push to GitHub

### 2.1 Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Tweet Sentiment Classifier"
```

### 2.2 Create GitHub Repository
1. Go to [github.com](https://github.com)
2. Click "New repository"
3. Name it: `tweet-sentiment-classifier`
4. Make it public
5. Don't initialize with README (you already have one)
6. Click "Create repository"

### 2.3 Push to GitHub
```bash
git remote add origin https://github.com/your-username/tweet-sentiment-classifier.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy to Streamlit Cloud

### 3.1 Sign Up for Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### 3.2 Deploy Your App
1. Click "New app"
2. Select your repository: `your-username/tweet-sentiment-classifier`
3. Branch: `main`
4. Main file path: `app.py`
5. Click "Deploy!"

### 3.3 Configure Environment Variables (Optional)
If you want to use environment variables:
1. Click "Advanced settings" before deploying
2. Add environment variable:
   - Key: `MODEL_NAME`
   - Value: `your-username/tweet-sentiment-classifier`

### 3.4 Wait for Deployment
- First deployment takes 2-5 minutes
- Streamlit will install dependencies from `requirements.txt`
- Model will be downloaded from Hugging Face Hub
- Your app will be available at: `https://your-app-name.streamlit.app`

---

## Step 4: Test Your Deployed App

1. Visit your app URL
2. Wait for model to load (first load takes ~30 seconds)
3. Test with example tweets
4. Share the URL!

---

## Troubleshooting

### App crashes during deployment
- Check logs in Streamlit Cloud dashboard
- Verify `requirements.txt` has all dependencies
- Ensure Hugging Face model repo is public

### Model not loading
- Verify model name in `app.py` matches your Hugging Face repo
- Check model repo is public on Hugging Face
- Check Streamlit Cloud logs for errors

### Out of memory errors
- Streamlit Cloud free tier has 1GB RAM
- Model should be ~256MB, which fits
- If issues persist, consider using a smaller model

### Slow loading
- First load downloads model from HF Hub (~30 seconds)
- Subsequent loads are cached
- Consider using Streamlit's caching decorators (already implemented)

---

## Alternative: Deploy Without Hugging Face Hub

If you prefer not to use Hugging Face Hub, you can:

### Option A: Use Git LFS (100 MB free tier)
```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git add models/
git commit -m "Add model with LFS"
git push
```

### Option B: Use Streamlit Secrets + Cloud Storage
1. Upload model to Google Drive/Dropbox/AWS S3
2. Add download logic in app.py
3. Use Streamlit secrets for credentials

---

## Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Hugging Face Hub Docs](https://huggingface.co/docs/hub)
- [Git LFS](https://git-lfs.com/)

---

## Quick Commands Summary

```bash
# 1. Upload model to HF
huggingface-cli login
huggingface-cli upload your-username/tweet-sentiment-classifier ./models/tweet-sentiment-classifier

# 2. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/tweet-sentiment-classifier.git
git push -u origin main

# 3. Deploy on Streamlit Cloud
# Go to share.streamlit.io and follow the GUI
```

---

**Your app will be live at:** `https://[app-name]-[random-id].streamlit.app`

Good luck with your deployment! 🚀
