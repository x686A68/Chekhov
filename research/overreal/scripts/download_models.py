"""Phase 1 — fetch the image stack (and Qwen3-32B) into the /data HF cache.

Runs in the background while Phase 0 proceeds. Logs one line per repo; failures are
recorded and do not stop the remaining downloads.
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

REPOS = [
    # (repo_id, allow_patterns) — smallest first so the image pilot can start early
    ("Qwen/Qwen2.5-VL-7B-Instruct", None),
    ("black-forest-labs/FLUX.1-dev", ["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*/*"]),
    ("Qwen/Qwen3-32B", None),
]


def main():
    for repo, patterns in REPOS:
        t0 = time.time()
        print(f"[start] {repo}", flush=True)
        try:
            path = snapshot_download(
                repo, allow_patterns=patterns, max_workers=8, ignore_patterns=["*.pth", "*.onnx", "*.bin"]
            )
            print(f"[done ] {repo} in {time.time()-t0:.0f}s -> {path}", flush=True)
        except Exception as e:  # noqa: BLE001 — record and continue
            print(f"[FAIL ] {repo} {type(e).__name__}: {str(e)[:300]}", flush=True)
    print("[all-done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
