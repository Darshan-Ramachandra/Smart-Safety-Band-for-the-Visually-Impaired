from pathlib import Path
import os

from huggingface_hub import snapshot_download


def download_repo(repo_id: str, local_dir: Path, token: str | None = None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )


def validate_transformers_dir(path: Path) -> None:
    required = ["config.json", "preprocessor_config.json"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing required files: {missing}")


def main() -> None:
    root = Path(__file__).resolve().parent / "weights"
    depth_dir = root / "depth-anything-small-hf"
    seg_dir = root / "segformer-b0-ade20k"
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    print("Downloading Depth Anything (small) ...")
    download_repo("LiheYoung/depth-anything-small-hf", depth_dir, token=token)
    validate_transformers_dir(depth_dir)
    print(f"Depth model ready at: {depth_dir}")

    print("Downloading SegFormer-B0 ADE20K ...")
    download_repo("nvidia/segformer-b0-finetuned-ade-512-512", seg_dir, token=token)
    validate_transformers_dir(seg_dir)
    print(f"SegFormer model ready at: {seg_dir}")

    print("All offline models are available.")


if __name__ == "__main__":
    main()
