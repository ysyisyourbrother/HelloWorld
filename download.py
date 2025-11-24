'''
HuggingFace Hub Downloader Script
This script downloads models or datasets from the HuggingFace Hub,
handling dependencies and authentication as needed.
Written by Ziwei
'''

import os
import sys
import subprocess
import logging
import argparse

BASE_DIR = "/mnt/share/cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --------------------------
# Logger setup
# --------------------------
logger = logging.getLogger("hf-downloader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


# --------------------------
# Dependency Check and Install
# --------------------------
try:
    import huggingface_hub
    logger.info("Module 'huggingface_hub' found.")
except ImportError:
    logger.info("Module 'huggingface_hub' not found. Attempting to install...")
    
    try:
        # 1. Ensure pip is available
        try:
            import pip
        except ImportError:
            logger.info("pip not found. Installing via ensurepip...")
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
            logger.info("pip installed.")
            
        # 2. Install/Upgrade huggingface_hub
        logger.info("Installing 'huggingface_hub'...")
        # Use -U and --force-reinstall for robustness against old versions
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"], check=True)
        logger.info("huggingface_hub installed successfully.")
        
        # 3. Final check
        import huggingface_hub
        
    except Exception as e:
        logger.error(f"Dependency setup failed: {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------
# Auto-detect repo type (model or dataset)
# seem not working, so we fallback to manual selection
# --------------------------
def detect_repo_type(repo_id: str) -> str:
    logger.info(f"Detecting repo type for '{repo_id}'...")
    try:
        info = huggingface_hub.get_repo_info(repo_id)
        if info.type == "model":
            logger.info("Auto-detected: model")
            return "model"
        elif info.type == "dataset":
            logger.info("Auto-detected: dataset")
            return "dataset"
        else:
            logger.warning(f"Unknown repo type returned: {info.type}")
    except Exception as e:
        logger.warning(f"Failed to detect repo type for '{repo_id}': {e}")

    # Manual selection if detection fails
    while True:
        choice = input(f"Cannot detect repo type for '{repo_id}'. Select manually [m]odel/[d]ataset: ").strip().lower()
        if choice in ("m", "model"):
            return "models"
        if choice in ("d", "dataset"):
            return "datasets"
        logger.error("Invalid input. Please type 'm' or 'd'.")


# --------------------------
# Download function
# --------------------------
def download_repo(repo_id: str, token: str | None, base_dir: str = BASE_DIR):
    # Login if token is provided
    if token:
        logger.info("Logging in using provided token...")
        try:
            huggingface_hub.login(token=token)
        except Exception as e:
            logger.error(f"Login failed: {e}")
    else:
        logger.info("No token provided. Proceeding without login.")

    # Detect repo type and set output directory
    repo_type = detect_repo_type(repo_id)
    out_dir = os.path.join(base_dir, repo_type, repo_id.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)

    # Download the repository
    logger.info(f"Downloading '{repo_id}' → {out_dir}")
    huggingface_hub.snapshot_download(repo_id=repo_id, local_dir=out_dir)
    logger.info(f"Download completed: {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Hub Downloader")
    parser.add_argument("--repo", required=True, help="Repo ID (model or dataset), e.g. google/gemma-2b")
    parser.add_argument("--token", required=False, help="Optional HuggingFace token")
    args = parser.parse_args()

    download_repo(args.repo, args.token)