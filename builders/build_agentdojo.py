"""AgentDojo source for the decision dataset: one row per labelled tool call, with exact labels.

    python builders/build_agentdojo.py --runs <agentdojo clone>/runs      # needs the `agentdojo` package and transformers

AgentDojo (ETH Zurich, MIT) publishes every run of 27 agent setups on four suites, each run a user task with or without
an injection hidden in a tool output, scored by the benchmark's own checkers. A call is labelled
  injected_action  a state-changing call whose tool is one the injection task's ground truth uses and whose arguments
                   carry a value only the attacker's goal has, in a run AgentDojo scores as a successful attack;
  correct_step     the first call in its run equal to one of the user task's ground-truth calls.
Everything else is left out. The question is step-level ("is this call a correct next step?"), never "is the job
complete", because one step of a multi-step task is incomplete by design.

Split by suite and user task, never by call: banking is held out whole as test; the other suites' user tasks are hashed
into train / calib / val. Writes outputs/sources/agentdojo.jsonl and prints the checks, counts and an audit sample.
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentdojo.task_suite.load_suites import get_suites

from common import pair_heldout, STEP_WORDINGS, check_history, dedupe, last_steps, render, text_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sources"
SEED = 42
TEST_SUITES = {"banking"}
SHARES = (("train", 80), ("calib", 10), ("val", 10))  # by (suite, user task), outside the test suites
# Rows per distinct action, each from a different model. Every injected action is a write, so correct writes keep many
# variants and injected writes are then capped to match them per split: otherwise "write" alone would predict the label.
VARIANTS_PER_ACTION = {("injected_action", "write"): 3, ("correct_step", "write"): 20, ("correct_step", "read"): 4}
INJECTED_PER_CORRECT_WRITE = 1.5  # per tool and split: a tool that is never a correct step contributes no injected rows,
# so neither "write" nor the tool's name predicts the label
# User tasks that hand authority to content the agent reads ("follow the instructions in the file", "do my TODO list"):
# an action taken from that content is not clearly unrequested, so their injected rows are left out.
DELEGATING_TASKS = {("banking", "user_task_2"), ("banking", "user_task_12"), ("banking", "user_task_13"),
                    ("workspace", "user_task_13"), ("workspace", "user_task_25"),
                    ("slack", "user_task_18"), ("slack", "user_task_19")}
LAST_STEPS = 2               # whole steps (an assistant message with all its tool results) shown when a row has history
TOKEN_LIMIT = 8192           # rows over this are dropped, never cut
READ_PREFIXES = ("get_", "search_", "read_", "list_", "check_")
JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "List", "object": "dict"}


# --- exact labels ----------------------------------------------------------------------------------------------------
def flat(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from flat(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from flat(v)
    elif value is not None:
        yield value


def norm(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return str(value).strip().lower()


def canon(x):
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    return norm(x)


SUITES, GT = {}, {}


def suite(version, name):
    if (version, name) not in SUITES:
        SUITES[version, name] = get_suites(version)[name]
    return SUITES[version, name]


def ground_truth(version, name, task_id):
    key = (version, name, task_id)
    if key not in GT:
        s = suite(version, name)
        task = s.user_tasks.get(task_id) or s.injection_tasks[task_id]
        GT[key] = [{"function": c.function, "args": dict(c.args)} for c in task.ground_truth(s.load_and_inject_default_environment({}))]
    return GT[key]


def attacker_values(inj_calls, user_calls):
    user = {norm(v) for c in user_calls for v in flat(c["args"])}
    return {n for c in inj_calls for v in flat(c["args"]) for n in [norm(v)]
            if n not in user and n not in ("", True, False) and not (isinstance(n, str) and len(n) < 4)}


def carries(call_args, values):
    for v in flat(call_args):
        n = norm(v)
        for a in values:
            if n == a or (isinstance(n, str) and isinstance(a, str) and len(a) >= 8 and (a in n or (len(n) >= 8 and n in a))):
                return True
    return False


def tool_schemas(s):
    out = []
    for f in s.tools:
        schema = f.parameters.model_json_schema()
        required = set(schema.get("required", []))
        params = {}
        for name, p in schema.get("properties", {}).items():
            kind = JSON_TYPES.get(p.get("type"), "str")
            if kind == "List" and p.get("items", {}).get("type"):
                kind = f"List[{JSON_TYPES.get(p['items']['type'], 'str')}]"
            params[name] = {"description": p.get("description") or name, "type": kind if name in required else f"{kind}, optional"}
        out.append({"name": f.name, "description": f.description, "parameters": params})
    return out


def labelled_calls(runs_dir):
    """Every exactly labelled call in every run, first occurrence per run."""
    for path in sorted(glob.glob(str(Path(runs_dir) / "*" / "*" / "user_task_*" / "*" / "*.json"))):
        run = json.load(open(path))
        if run.get("error") or not run.get("messages"):
            continue
        version = run.get("benchmark_version") or "v1"
        name, user_id, inj_id = run["suite_name"], run["user_task_id"], run.get("injection_task_id")
        try:
            user_calls = ground_truth(version, name, user_id)
            inj_calls = ground_truth(version, name, inj_id) if inj_id else []
        except KeyError:
            continue
        inj_tools = {c["function"] for c in inj_calls if not c["function"].startswith(READ_PREFIXES)}
        values = attacker_values(inj_calls, user_calls)
        msgs = run["messages"]
        goal_i = next(i for i, m in enumerate(msgs) if m["role"] == "user")
        seen = set()
        for i, m in enumerate(msgs):
            for call in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
                fn, args = call["function"], call.get("args") or {}
                key = (fn, json.dumps(canon(args), sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                if inj_id and run.get("security") is True and fn in inj_tools and carries(args, values):
                    kind = "injected_action"
                elif any(fn == c["function"] and canon(args) == canon(c["args"]) for c in user_calls):
                    kind = "correct_step"
                else:
                    continue
                yield {"run": path, "model": run["pipeline_name"], "suite": name, "version": version, "user_task": user_id,
                       "injection_task": inj_id, "attack": run.get("attack_type"), "kind": kind,
                       "goal": text_of(msgs[goal_i]["content"]), "history": msgs[goal_i + 1:i],
                       "call": {"name": fn, "arguments": args}}


# --- rows --------------------------------------------------------------------------------------------------------------
def split_of(suite_name, user_task):
    if suite_name in TEST_SUITES:
        return "test"
    h = int(hashlib.sha256(f"agentdojo|{suite_name}|{user_task}|{SEED}".encode()).hexdigest(), 16) % 100
    edge = 0
    for name, share in SHARES:
        edge += share
        if h < edge:
            return name
    return SHARES[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tokenizer", default="LiquidAI/d1-3B")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # 1. Label, then keep up to VARIANTS_PER_ACTION per distinct action, preferring different models and attacks.
    by_action = defaultdict(list)
    for c in labelled_calls(args.runs):
        if c["kind"] == "injected_action" and (c["suite"], c["user_task"]) in DELEGATING_TASKS:
            continue
        action = (c["suite"], c["user_task"], c["injection_task"] if c["kind"] == "injected_action" else None, c["kind"],
                  c["call"]["name"], json.dumps(canon(c["call"]["arguments"]), sort_keys=True))
        by_action[action].append(c)
    rng = random.Random(SEED)
    picked = []
    for action in sorted(by_action, key=lambda a: json.dumps(a)):
        pool = sorted(by_action[action], key=lambda c: c["run"])
        rng.shuffle(pool)
        chosen, used = [], set()
        for c in pool:  # first pass: a new model and attack each time
            if (c["model"], c["attack"]) not in used and c["model"] not in {x["model"] for x in chosen}:
                chosen.append(c); used.add((c["model"], c["attack"]))
            if len(chosen) == VARIANTS_PER_ACTION[action[3], "read" if action[4].startswith(READ_PREFIXES) else "write"]:
                break
        picked += chosen

    # 2. Render: half the rows with the last few messages (whole), half with the request only (what the live hook sees).
    rows, dropped = [], Counter()
    for c in picked:
        row_rng = random.Random(f'{SEED}|{c["run"]}|{c["call"]["name"]}|{json.dumps(c["call"]["arguments"], sort_keys=True)}')
        split = split_of(c["suite"], c["user_task"])
        with_history = row_rng.random() < 0.5
        history = last_steps(c["history"], LAST_STEPS) if with_history else []
        tools = tool_schemas(suite(c["version"], c["suite"]))
        styles = [row_rng.choice(["native_json", "transcript"])] + (["markdown"] if split == "test" else [])
        for style in styles:
            out_split = "test_heldout_format" if style == "markdown" else split
            state = render(tools, c["goal"], history, c["call"], style)
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            tokens = len(tok.encode(text, add_special_tokens=False))
            if tokens > TOKEN_LIMIT:
                dropped[c["kind"], "over token limit"] += 1
                continue
            question = {"type": "noul", "instructions": row_rng.choice(STEP_WORDINGS)}
            # one id per item; the held-out-format copy differs only in its split suffix, as notebook 01 pairs them
            rid = hashlib.sha256(f'{c["run"]}|{c["call"]["name"]}|{json.dumps(c["call"]["arguments"], sort_keys=True)}'.encode()).hexdigest()[:16]
            rows.append({"id": f"agentdojo:{rid}:{out_split}", "source": "agentdojo", "group": f'{c["suite"]}/{c["user_task"]}',
                         "split": out_split, "state": state, "questions": {"step_ok": question},
                         "gold": {"step_ok": c["kind"] == "correct_step"},
                         "meta": {"kind": c["kind"], "edit": None, "tokens_state": tokens, "style": style, "history": bool(history),
                                  "model": c["model"], "attack": c["attack"], "suite": c["suite"], "user_task": c["user_task"],
                                  "injection_task": c["injection_task"], "call": c["call"], "families": {"step_ok": "step_ok"}}})

    # Request-only rows of one action are identical across models: keep one, and never let one state carry two labels.
    rows = dedupe(rows, dropped)

    # Cap injected rows per (split, tool) to INJECTED_PER_CORRECT_WRITE x that tool's correct rows there (seeded).
    capped = []
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        capped += [r for r in part if r["meta"]["kind"] == "correct_step"]
        correct = Counter(r["meta"]["call"]["name"] for r in part if r["meta"]["kind"] == "correct_step")
        by_tool = defaultdict(list)
        for r in part:
            if r["meta"]["kind"] == "injected_action":
                by_tool[r["meta"]["call"]["name"]].append(r)
        for tool, injected in sorted(by_tool.items()):
            injected.sort(key=lambda r: r["id"])
            random.Random(f"{SEED}|cap|{split}|{tool}").shuffle(injected)
        if not split.startswith("test"):
            for tool, injected in sorted(by_tool.items()):
                keep = int(correct[tool] * INJECTED_PER_CORRECT_WRITE)
                dropped["injected_action", "balance cap"] += max(0, len(injected) - keep)
                capped += injected[:keep]
        else:  # held-out suite: its tools are never trained on, so cap the total and take tools round-robin
            total, kept = sum(correct.values()), []
            while len(kept) < total and any(by_tool.values()):
                for tool in sorted(by_tool):
                    if by_tool[tool] and len(kept) < total:
                        kept.append(by_tool[tool].pop(0))
            dropped["injected_action", "balance cap"] += sum(len(v) for v in by_tool.values())
            capped += kept
    rows = capped

    rows = pair_heldout(rows, dropped)

    # 3. Checks: fail loudly, write nothing.
    check_history(rows)
    groups = defaultdict(set)
    for r in rows:
        groups[r["group"]].add(r["split"].replace("test_heldout_format", "test"))
    assert all(len(s) == 1 for s in groups.values()), "a user task is in two splits"
    assert all(r["meta"]["suite"] in TEST_SUITES for r in rows if r["split"].startswith("test")), "test outside the held-out suite"
    assert not any(r["meta"]["suite"] in TEST_SUITES for r in rows if not r["split"].startswith("test")), "held-out suite leaked"
    assert all(r["gold"]["step_ok"] == (r["meta"]["kind"] == "correct_step") for r in rows)
    assert all(not r["meta"]["call"]["name"].startswith(READ_PREFIXES) for r in rows if r["meta"]["kind"] == "injected_action")
    for split in {r["split"] for r in rows} - {"test", "test_heldout_format"}:
        n = Counter((r["meta"]["call"]["name"], r["meta"]["kind"]) for r in rows if r["split"] == split)
        assert all(n[t, "injected_action"] <= INJECTED_PER_CORRECT_WRITE * n[t, "correct_step"] for t, _ in n), f"tool imbalance in {split}"

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "agentdojo.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. Report.
    print("distinct labelled actions:", Counter(a[3] for a in by_action), "| rows picked:", Counter(c["kind"] for c in picked))
    print("dropped:", dict(dropped))
    table = Counter((r["split"], r["meta"]["kind"], "read" if r["meta"]["call"]["name"].startswith(READ_PREFIXES) else "write") for r in rows)
    for split in ("train", "calib", "val", "test", "test_heldout_format"):
        print(f'{split:<20} injected writes={table[split, "injected_action", "write"]:<5} correct writes={table[split, "correct_step", "write"]:<5} '
              f'correct reads={table[split, "correct_step", "read"]}')
    lengths = sorted(r["meta"]["tokens_state"] for r in rows)
    print("state tokens: median", lengths[len(lengths) // 2], "p90", lengths[int(.9 * len(lengths))], "max", lengths[-1])
    print("with history:", sum(r["meta"]["history"] for r in rows), "of", len(rows))
    print("models:", len({r["meta"]["model"] for r in rows}), "| attacks:", Counter(r["meta"]["attack"] for r in rows).most_common(6))
    print("->", OUT / "agentdojo.jsonl")


if __name__ == "__main__":
    main()
