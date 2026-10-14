"""Log outputs/sources/*.jsonl to W&B as the `decision-sources` artifact, with a manifest of where each came from.

    python builders/log_sources.py --agentdojo <clone> --tau <clone> --bfcl <dir> --halueval <csv>

Run it from outside the repo root (the repo's wandb/ run folder shadows the package). Notebook 01 downloads a pinned
version of this artifact, encodes the rows for its target model and runs its hard checks on them.
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "outputs" / "sources"
FILES = ["agentdojo.jsonl", "tau_bench.jsonl", "bfcl.jsonl", "grounding.jsonl"]


def env(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agentdojo", required=True)
    ap.add_argument("--tau", required=True)
    ap.add_argument("--bfcl", required=True)
    ap.add_argument("--halueval", required=True)
    args = ap.parse_args()
    import wandb

    counts = {}
    for name in FILES:
        rows = [json.loads(line) for line in open(SOURCES / name)]
        counts[name] = {f"{s}/{k}": n for (s, k), n in sorted(Counter((r["split"], r["meta"]["kind"]) for r in rows).items())}
    manifest = {
        "builders_commit": git_head(ROOT),
        "upstream": {
            "agentdojo": {"repo": "https://github.com/ethz-spylab/agentdojo", "commit": git_head(args.agentdojo), "license": "MIT"},
            "tau_bench": {"repo": "https://github.com/sierra-research/tau-bench", "commit": git_head(args.tau), "license": "MIT"},
            "bfcl": {"repo": "https://github.com/ShishirPatil/gorilla", "commit": Path(args.bfcl, "REVISION").read_text().strip(),
                     "license": "Apache-2.0"},
            "halueval": {"file": Path(args.halueval).name, "sha256": sha256(args.halueval), "license": "MIT",
                         "origin": "HaluEval (QA from HotpotQA, dialogue from OpenDialKG), exported by the earlier grounded-ai judges"},
        },
        "files": {name: {"sha256": sha256(SOURCES / name), "rows": sum(counts[name].values()), "by_split_kind": counts[name]}
                  for name in FILES},
    }
    (SOURCES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    keys = env(ROOT / ".env")
    wandb.login(key=keys["WANDB_API_KEY"], relogin=True)
    run = wandb.init(project="grounded-decision", job_type="sources", config={"files": FILES})
    art = wandb.Artifact("decision-sources", type="dataset", metadata=manifest,
                         description="AgentDojo, tau-bench, BFCL and HaluEval-grounding rows with exact labels (grounded-decision builders/)")
    for name in FILES + ["manifest.json"]:
        art.add_file(str(SOURCES / name))
    logged = run.log_artifact(art)
    logged.wait()
    print("logged", f"decision-sources:{logged.version}", "| builders commit", manifest["builders_commit"])
    run.finish()


if __name__ == "__main__":
    main()
