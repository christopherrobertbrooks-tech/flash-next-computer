"""Stage 0 provenance (Codex finding 3): sha256 of every artifact the stage used, plus the llama.cpp commit, in one
file the gate can verify later (stage0_gate.py --provenance). Refuses to overwrite.

usage: python provenance.py OUT.json
"""
import hashlib, json, subprocess, sys
from pathlib import Path

L = Path("/home/chris/llama.cpp-mainline")
W = Path("/mnt/nvme/flashnext-test/llamacpp")
GROUPS = {
    "qwenputer checkpoint": sorted(Path("/mnt/nvme/qwenputer/model").glob("*.safetensors")),
    "plain checkpoint": sorted(Path("/mnt/nvme/qwen35-4b/model").glob("*.safetensors")),
    "qwen half (cut)": sorted((W / "qwen-half").glob("*")),
    "ggufs": [W / "qwen-half-f16.gguf", Path("/mnt/nvme/qwen35-4b/gguf/qwen35-4b-f16.gguf")],
    "helper and llama.cpp build": [W / "qwen_half_helper", W / "qwen_half_helper.cpp",
                                   *sorted((L / "build-sm70/bin").glob("lib*.so.0.*")),
                                   L / "build-sm70/bin/llama-quantize", L / "convert_hf_to_gguf.py",
                                   L / "conversion/qwen.py"],
    "stage 0 code": sorted(W.glob("*.py")) + [W / "stage0d.sh"],
}


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
    files, groups = {}, {}
    for g, paths in GROUPS.items():
        groups[g] = [str(p) for p in paths]
        for p in paths:
            files[str(p)] = sha(p)
            print(f"{files[str(p)][:12]}  {p}", flush=True)
    commit = subprocess.run(["git", "-C", str(L), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out.write_text(json.dumps({"llama_cpp_commit": commit, "groups": groups, "files": files}, indent=1) + "\n")


if __name__ == "__main__":
    main()
