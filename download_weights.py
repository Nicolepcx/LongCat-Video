#!/usr/bin/env python3
"""
Download LongCat-Video model weights to a local directory.

This is intended to be run **once** on a RunPod pod (or any machine) that has
the target Network Volume attached.  After the download the serverless workers
will read the weights at startup without needing to re-download.

Usage
-----
    python download_weights.py                           # uses default /runpod-volume/weights
    python download_weights.py --output_dir ./weights    # custom path

The script downloads two HuggingFace repos:

    meituan-longcat/LongCat-Video        → <output_dir>/LongCat-Video/
    meituan-longcat/LongCat-Video-Avatar  → <output_dir>/LongCat-Video-Avatar/

Set the HF_TOKEN environment variable if the repos are gated.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download LongCat-Video model weights from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("WEIGHTS_DIR", "/runpod-volume/weights"),
        help="Root directory to store weights (default: /runpod-volume/weights or $WEIGHTS_DIR)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repos = [
        ("meituan-longcat/LongCat-Video", "LongCat-Video"),
        ("meituan-longcat/LongCat-Video-Avatar", "LongCat-Video-Avatar"),
    ]

    for repo_id, folder_name in repos:
        dest = output_dir / folder_name
        print(f"\n{'='*60}")
        print(f"Downloading {repo_id}")
        print(f"  → {dest}")
        print(f"{'='*60}\n")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=args.token,
        )

        print(f"\n✓ {repo_id} downloaded to {dest}")

    print(f"\n{'='*60}")
    print(f"All weights downloaded to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
