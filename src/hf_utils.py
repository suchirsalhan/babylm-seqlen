# src/hf_utils.py
from huggingface_hub import HfApi, create_repo, upload_folder

def save_to_hf(model_type, local_path, checkpointing_config, step):
    repo_id = checkpointing_config.hf_checkpoint.repo_id or f"babylm-seqlen/{model_type}-trained"
    
    # Create the repo if it doesn't exist (can be skipped if repo already exists)
    create_repo(repo_id, exist_ok=True)
    
    # Create a new branch for each checkpoint (e.g., 'checkpoint-{step}')
    branch_name = f"checkpoint-{step}"
    
    # Upload the model to the created branch
    upload_folder(
        repo_id=repo_id,
        folder_path=local_path,
        commit_message=f"Upload model checkpoint {model_type} at step {step}",
        repo_type="model",
        branch=branch_name  # Specify the branch
    )
    
    print(f"✅ Uploaded checkpoint to Hugging Face Hub: https://huggingface.co/{repo_id}/tree/{branch_name}")
