"""FAST-PLAN: write one arm's frozen runtime identity (sha256 of every file the run uses), checked by run_split.py
--expect at startup. Arm F = the stock Qwen3.5-4B Q8_0 GGUF, arm G = the f16 GGUF; both with qwen_half_helper (token
mode) and upstream's executor. Also records the upstream qwenputer and llama.cpp commits and checks text-id
compatibility (the base tokenizer equals Qwenputer's on every text id, its unused reserved slots aside). Refuses to
overwrite.

usage (on the gateway): python make_fast_artifacts.py F|G OUT.json
"""
import hashlib, json, subprocess, sys
from pathlib import Path

W = Path("/mnt/nvme/flashnext-test/llamacpp")
L = Path("/home/chris/llama.cpp-mainline")
UP = Path("/mnt/nvme/qwenputer/qwenputer")
RV = UP / "experimental" / "riscv"
M = Path("/mnt/nvme/qwenputer/model")
BASE = Path("/mnt/nvme/qwen35-4b/model")
GGUF = {"F": W / "qwen35-4b-q8_0.gguf", "G": Path("/mnt/nvme/qwen35-4b/gguf/qwen35-4b-f16.gguf")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def text_ids_compatible():
    q = json.loads((M / "tokenizer.json").read_text())
    b = json.loads((BASE / "tokenizer.json").read_text())
    qm = {i: t for t, i in q["model"]["vocab"].items()} | {a["id"]: a["content"] for a in q["added_tokens"]}
    bm = {i: t for t, i in b["model"]["vocab"].items()} | {a["id"]: a["content"] for a in b["added_tokens"]}
    v0 = json.loads((M / "chipformer.json").read_text())["chip_token_offset"]
    diff = [i for i in range(v0) if qm.get(i) != bm.get(i)
            and not (bm.get(i) is None and str(qm.get(i)).startswith("<|reserved_"))]
    return not diff and max(bm) < v0


def main():
    arm, out = sys.argv[1], Path(sys.argv[2])
    if out.exists():
        sys.exit(f"{out} exists")
    assert text_ids_compatible(), "the base tokenizer differs from Qwenputer's on a text id"
    emb = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]["model.embed_tokens.weight"]
    paths = [GGUF[arm], W / "qwen_half_helper", W / "qwen_half_helper.cpp",
             *sorted((L / "build-sm70/bin").glob("lib*.so.0.*")),
             W / "run_split.py", W / "split_chat.py", W / "helper_client.py", W / "stage0_gate.py",
             W / "run_helper_fixtures.py", W / "fast.sh", W / "inside4b/run_qwenputer.py", W / "inside4b/frames.py",
             W / "final-primary.jsonl", W / "fixtures-v1/manifest.json", W / "plain-hf-v2.json", W / "plain-llama.json",
             M / "tokenizer.json", M / "tokenizer_config.json", M / "config.json", M / "chipformer.json",
             M / "chat_template.jinja", M / "model.safetensors.index.json", M / emb,
             BASE / "tokenizer.json", BASE / "tokenizer_config.json", BASE / "config.json",
             RV / "tests/data/rv/calc-bridge.elf", *sorted((RV / "chipformer").rglob("*.py"))]
    files = {str(p): sha(p) for p in paths}
    git = lambda d, *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True).stdout.strip()
    rec = {"arm": arm, "gguf": str(GGUF[arm]), "lm_input": "tokens", "files": files, "text_ids_compatible": True,
           "qwenputer_commit": git(UP, "rev-parse", "HEAD"),
           "qwenputer_dirty": git(UP, "status", "--porcelain", "--untracked-files=no"),
           "llama_cpp_commit": git(L, "rev-parse", "HEAD")}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"arm {arm}: {len(files)} files; qwenputer {rec['qwenputer_commit'][:12]}; llama.cpp {rec['llama_cpp_commit'][:12]}")


if __name__ == "__main__":
    main()
