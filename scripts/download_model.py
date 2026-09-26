"""Download registry models: python scripts/download_model.py qwen2.5-1.5b [more keys...] | --list"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess  # noqa: E402

from huggingface_hub import HfApi, hf_hub_url  # noqa: E402

from llev.config import get_settings  # noqa: E402
from llev.models import REGISTRY, local_path  # noqa: E402


def resolve(repo: str, pattern: str) -> str:
    files = [f for f in HfApi().list_repo_files(repo) if f.lower().endswith(".gguf")]
    matches = sorted(f for f in files if pattern.lower() in f.lower() and "/" not in f)
    if not matches:
        raise SystemExit(f"{repo}: no .gguf matching {pattern!r}; available: {files}")
    return min(matches, key=len)  # the plain file, not split/variant ones


def main(keys: list[str]) -> None:
    s = get_settings()
    if not keys or keys == ["--list"]:
        for k, m in REGISTRY.items():
            have = "✓" if local_path(s.models_dir, k) else " "
            print(f"[{have}] {k:14} {m.params:5} {m.repo}  — {m.note}")
        return
    for key in keys:
        spec = REGISTRY.get(key) or sys.exit(f"unknown model {key!r}; see --list")
        if (p := local_path(s.models_dir, key)) is not None:
            print(f"{key}: already at {p}")
            continue
        fname = resolve(spec.repo, spec.pattern)
        print(f"{key}: downloading {spec.repo}/{fname}")
        dest = Path(s.models_dir) / key
        dest.mkdir(parents=True, exist_ok=True)
        part = dest / (fname + ".part")
        # curl resumes (-C -) and retries on stalls; more robust than hf_hub on slow links.
        subprocess.run(["curl", "-fL", "--retry", "30", "--retry-all-errors", "--retry-delay", "3",
                        "--speed-limit", "10000", "--speed-time", "30", "-C", "-", "-sS",
                        "-o", str(part), hf_hub_url(spec.repo, fname)], check=True)
        part.rename(dest / fname)
        print(f"{key}: {dest / fname}")


if __name__ == "__main__":
    main(sys.argv[1:])
