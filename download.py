from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="gpt2",
    local_dir="gpt2_local",
    local_dir_use_symlinks=False,
)
print("GPT-2 downloaded to gpt2_local/")
