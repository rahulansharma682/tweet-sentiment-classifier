"""
Helper script to upload your model to Hugging Face Hub
Run this after logging in with: huggingface-cli login
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration - EDIT THIS!
USERNAME = "ranashch"  # Replace with your Hugging Face username
MODEL_REPO_NAME = "tweet-sentiment-classifier"
LOCAL_MODEL_PATH = "./models/tweet-sentiment-classifier"

def upload_model():
    """Upload model to Hugging Face Hub"""

    # Check if username was changed
    if USERNAME == "your-username":
        print("❌ ERROR: Please edit this file and replace 'your-username' with your Hugging Face username!")
        print("   Edit line 9 in upload_to_huggingface.py")
        return

    # Check if model exists
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ ERROR: Model directory not found at: {LOCAL_MODEL_PATH}")
        print("   Please ensure your trained model is in the models/ directory")
        return

    repo_id = f"{USERNAME}/{MODEL_REPO_NAME}"

    print("=" * 60)
    print("🤗 Uploading Model to Hugging Face Hub")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Local path: {LOCAL_MODEL_PATH}")
    print()

    # Create repository
    api = HfApi()
    try:
        print("📦 Creating repository...")
        create_repo(repo_id=repo_id, repo_type="model", private=False)
        print(f"✅ Repository created successfully!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Repository already exists, will update it")
        else:
            print(f"❌ Error creating repository: {e}")
            return

    # Upload files
    try:
        print()
        print("📤 Uploading model files... (this may take a few minutes)")
        api.upload_folder(
            folder_path=LOCAL_MODEL_PATH,
            repo_id=repo_id,
            repo_type="model"
        )
        print()
        print("=" * 60)
        print("✅ SUCCESS! Model uploaded to Hugging Face Hub")
        print("=" * 60)
        print()
        print(f"🔗 View your model at: https://huggingface.co/{repo_id}")
        print()
        print("📝 NEXT STEPS:")
        print(f"   1. Edit app.py line 21 and change MODEL_NAME to: \"{repo_id}\"")
        print("   2. Commit and push your code to GitHub")
        print("   3. Deploy on Streamlit Cloud")
        print()
        print("   See DEPLOYMENT.md for detailed instructions")
        print()

    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        print()
        print("💡 Make sure you've logged in with: huggingface-cli login")
        return

if __name__ == "__main__":
    print()
    print("🚀 Hugging Face Model Upload Script")
    print()

    # Check if logged in
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"✅ Logged in as: {whoami['name']}")
        print()
    except Exception:
        print("❌ Not logged in to Hugging Face!")
        print()
        print("Please run: huggingface-cli login")
        print("Then run this script again.")
        print()
        exit(1)

    upload_model()
