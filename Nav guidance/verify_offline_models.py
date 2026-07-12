from pathlib import Path


def check_dir(path: Path, required_files: list[str]) -> list[str]:
    missing = []
    for f in required_files:
        if not (path / f).exists():
            missing.append(f)
    return missing


def main() -> None:
    root = Path(__file__).resolve().parent / "weights"
    depth_dir = root / "depth-anything-small-hf"
    seg_dir = root / "segformer-b0-ade20k"
    yolo_file = root / "yolo11n.pt"

    print(f"Checking weights root: {root}")
    if not yolo_file.exists():
        print(f"[MISSING] {yolo_file}")
    else:
        print(f"[OK] {yolo_file}")

    depth_missing = check_dir(
        depth_dir,
        ["config.json", "preprocessor_config.json"],
    )
    if depth_missing:
        print(f"[MISSING] {depth_dir} missing: {depth_missing}")
    else:
        print(f"[OK] {depth_dir}")

    seg_missing = check_dir(
        seg_dir,
        ["config.json", "preprocessor_config.json"],
    )
    if seg_missing:
        print(f"[MISSING] {seg_dir} missing: {seg_missing}")
    else:
        print(f"[OK] {seg_dir}")


if __name__ == "__main__":
    main()

