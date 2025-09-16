import argparse
from huggingface_hub import snapshot_download
HF_TOKEN = "REMOVEDqrbGlArMBwHjsgWWvtOtMkHmRfAmLuNqXj"

def download(repo, types, save_dir):
    print(f"Downloading {repo} ({types}) to {save_dir}...")
    snapshot_download(
        repo_id=repo,
        repo_type=types,
        local_dir=save_dir,
        token=HF_TOKEN
    )
    print("Download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets/models from Hugging Face")
    parser.add_argument("--repo", default="THP2903/BV175_COT_Xvi_2", type=str, required=False, help="Hugging Face repo ID (e.g., 'THP2903/x_ray_8bit')")
    parser.add_argument("--type", default="dataset", type=str, choices=["dataset", "model"], required=False, help="Type: 'dataset' or 'model'")
    parser.add_argument("--save_dir", default="/home/datnvt/project/Medical_CLARA/data_detection", type=str, required=False, help="Directory to save downloaded files")

    args = parser.parse_args()
    download(args.repo, args.type, args.save_dir)
