"""LLAMACPP-PLAN v3 stage 3: write the frozen artifact list (sha256 of every file the run uses) that run_split.py
--expect checks at startup. Also records the upstream qwenputer commit and whether its tree is clean, and the
llama.cpp commit. Refuses to overwrite.

usage (on the gateway): python make_stage3_artifacts.py OUT.json
"""
import hashlib, json, subprocess, sys
from pathlib import Path

W = Path("/mnt/nvme/flashnext-test/llamacpp")
L = Path("/home/chris/llama.cpp-mainline")
UP = Path("/mnt/nvme/qwenputer/qwenputer")
RV = UP / "experimental" / "riscv"
M = Path("/mnt/nvme/qwenputer/model")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    if out.exists():
        sys.exit(f"{out} exists")
    emb = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]["model.embed_tokens.weight"]
    paths = [W / "qwen-half-f16-bf16head.gguf", W / "qwen_half_helper", W / "qwen_half_helper.cpp",
             *sorted((L / "build-sm70/bin").glob("lib*.so.0.*")),
             W / "run_split.py", W / "split_chat.py", W / "helper_client.py", W / "stage0_gate.py",
             W / "run_helper_fixtures.py", W / "inside4b/run_qwenputer.py", W / "inside4b/frames.py",
             W / "final-primary.jsonl", W / "fixtures-v1/manifest.json", W / "plain-hf-v2.json", W / "plain-llama.json",
             M / "tokenizer.json", M / "tokenizer_config.json", M / "config.json", M / "chipformer.json",
             M / "chat_template.jinja", M / "model.safetensors.index.json", M / emb,
             RV / "tests/data/rv/calc-bridge.elf", *sorted((RV / "chipformer").rglob("*.py"))]
    files = {str(p): sha(p) for p in paths}
    git = lambda d, *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True).stdout.strip()
    rec = {"files": files, "qwenputer_commit": git(UP, "rev-parse", "HEAD"),
           "qwenputer_dirty": git(UP, "status", "--porcelain", "--untracked-files=no"),
           "llama_cpp_commit": git(L, "rev-parse", "HEAD")}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{len(files)} files; qwenputer {rec['qwenputer_commit'][:12]} (dirty: {bool(rec['qwenputer_dirty'])}); "
          f"llama.cpp {rec['llama_cpp_commit'][:12]}")


if __name__ == "__main__":
    main()
