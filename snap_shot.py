from huggingface_hub import snapshot_download
import os

def download_models(models, base_dir, token, type="model"):
    """
    Download multiple models from Hugging Face Hub to separate folders.

    Args:
        models (dict): Dictionary where keys are model repo_ids and values are folder names.
        base_dir (str): Base directory to store downloaded models.
        token (str): Hugging Face API token.
    """
    os.makedirs(base_dir, exist_ok=True)

    for repo_id, folder_name in models.items():
        local_path = os.path.join(base_dir, folder_name)
        print(f"Downloading {repo_id} to {local_path}...")
        
        snapshot_download(
            repo_id=repo_id,
            repo_type=type,
            local_dir=local_path,
            token=token
        )
        
        print(f"Downloaded {repo_id} successfully!\n")

if __name__ == "__main__":
    models_to_download = {
        # "Viet-Mistral/Vistral-7B-Chat": "Vistral-7B-Chat",
        # "Vi-VLM/Vistral-V-7B": "Vistral-V-7B",
        # "Vi-VLM/llava-vistral-7b-lora": "llava-vistral-7b-lora",
        # "Vi-VLM/llava-vistral-7b-pretrain": "llava-vistral-7b-pretrain",
        # "5CD-AI/Vintern-1B-v2": "Vintern-1B-v2",
        # "5CD-AI/Vintern-1B-v3_5": "Vintern-1B-v3_5",
        # "5CD-AI/Vintern-3B-beta": "Vintern-3B-beta",
        # "THP2903/Qwen2-VL-7B-Instruct_finding_full" : "Qwen2-VL-7B-Instruct_finding_full",
        # "pythera/vimed175","data"
        # "THP2903/Qwen2vl_instruct_medical_2": "clara",
        # "THP2903/Qwen2vl_7b_instruct_medical_multiturn_full": "clara_multiturn",
        # "chitb/LaVy-pretrain": "lavy-pretrain",
        # "THP2903/xray_detection_mae" :"xray_detection_mae",
        # "THP2903/x_ray_8bit":"x_ray_8bit",
        
    }
    base_directory = r"/home/datnvt/project/data/data_mae"
    REMOVEDtoken = "REMOVED"
    type="dataset"  # Change to "model" if downloading models
    download_models(models_to_download, base_directory, REMOVEDtoken, type)
