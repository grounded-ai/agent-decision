"""Shared by the trace sources (AgentDojo, tau-bench): message views, rendering, splits, the step question, checks.

A trace row shows the tools, what the user asked, optionally the last whole steps before the call, and the proposed call.
"""
import hashlib
import json
import random
from collections import Counter

JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "List", "object": "dict",
              "dict": "dict", "float": "float", "tuple": "tuple", "any": "Any"}  # the last four are BFCL's own type names
STEP_WORDINGS = ["Is this tool call a correct next step toward what the user asked?",
                 "Should this tool call be executed as proposed?",
                 "Is the assistant's proposed call a correct step for the user's request, with the right arguments?"]


def xlam_tool(name, description, schema):
    """A JSON-schema tool in the xLAM format the models were trained on."""
    required = set(schema.get("required", []))
    params = {}
    for p_name, p in schema.get("properties", {}).items():
        kind = JSON_TYPES.get(p.get("type"), "str")
        if kind == "List" and p.get("items", {}).get("type"):
            kind = f"List[{JSON_TYPES.get(p['items']['type'], 'str')}]"
        params[p_name] = {"description": p.get("description") or p_name, "type": kind if p_name in required else f"{kind}, optional"}
    return {"name": name, "description": description, "parameters": params}


def text_of(content):
    if isinstance(content, list):
        return "\n".join(x.get("content", "") or x.get("text", "") for x in content if isinstance(x, dict))
    return content or ""


def call_of(c):
    """A tool call from either AgentDojo ({function, args}) or OpenAI ({function: {name, arguments}}) messages."""
    if isinstance(c.get("function"), dict):
        args = c["function"].get("arguments")
        return {"name": c["function"]["name"], "arguments": args if isinstance(args, dict) else json.loads(args or "{}")}
    return {"name": c["function"], "arguments": c.get("args") or {}}


def view(m):
    out = {"role": m["role"]}
    if m["role"] == "tool":
        name = m["tool_call"].get("function") if isinstance(m.get("tool_call"), dict) else m.get("name")
        if name:
            out["tool"] = name
    if text_of(m.get("content")):
        out["content"] = text_of(m.get("content"))
    if m.get("tool_calls"):
        out["tool_calls"] = [call_of(c) for c in m["tool_calls"]]
    return out


def last_steps(history, n):
    """The last n whole steps: each starts at an assistant or user message and keeps every tool result after it."""
    starts = [i for i, m in enumerate(history) if m["role"] != "tool"]
    if not starts:
        return []
    return history[starts[-n]:] if len(starts) >= n else history[starts[0]:]


def render(tools, request, history, call, style):
    history = [view(m) for m in history]
    if style == "native_json":
        state = {"tools": tools, "user": request}
        if history:
            state["history"] = history
        state["assistant_tool_calls"] = [call]
        return state

    def line(m):
        if m["role"] == "tool":
            name = f' ({m["tool"]})' if m.get("tool") else ""
            return f'TOOL RESULT{name}: {m.get("content", "")}' if style == "transcript" else f'**Tool result{name}**\n```\n{m.get("content", "")}\n```'
        calls = "\n".join(f'-> {c["name"]}({json.dumps(c["arguments"], ensure_ascii=False)})' for c in m.get("tool_calls", []))
        text = "\n".join(x for x in (m.get("content", ""), calls) if x)
        who = {"assistant": "ASSISTANT", "user": "USER"}.get(m["role"], m["role"].upper())
        return f"{who}: {text}" if style == "transcript" else f"**{who.title()}**\n{text}"

    proposed = f'-> {call["name"]}({json.dumps(call["arguments"], ensure_ascii=False)})'
    if style == "transcript":
        parts = ["TOOLS:\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools), f"USER: {request}"]
        if history:
            parts.append("EARLIER STEPS:\n" + "\n".join(line(m) for m in history))
        return "\n\n".join(parts + [f"ASSISTANT (proposed tool call):\n{proposed}"])
    tool_lines = "\n".join(f'- `{t["name"]}`: {t["description"]} Parameters: {json.dumps(t["parameters"], ensure_ascii=False)}' for t in tools)
    parts = [f"## Available tools\n{tool_lines}", f"## User request\n{request}"]
    if history:
        parts.append("## Earlier steps\n" + "\n\n".join(line(m) for m in history))
    return "\n\n".join(parts + [f"## Proposed tool call\n```\n{proposed}\n```"])


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def hashed_split(key, shares, seed):
    h = int(hashlib.sha256(f"{key}|{seed}".encode()).hexdigest(), 16) % 100
    edge = 0
    for name, share in shares:
        edge += share
        if h < edge:
            return name
    return shares[-1][0]


def step_question(rng):
    return {"type": "noul", "instructions": rng.choice(STEP_WORDINGS)}


def dedupe(rows, dropped):
    """Keep one row per (split, state); one state must never carry two labels."""
    gold_of, unique = {}, []
    for r in rows:
        k = (r["split"], r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], sort_keys=True))
        if k in gold_of:
            assert gold_of[k] == r["gold"]["step_ok"], f"one state, two labels: {r['id']}"
            dropped[r["meta"]["kind"], "duplicate state"] += 1
            continue
        gold_of[k] = r["gold"]["step_ok"]
        unique.append(r)
    return unique


def check_history(rows):
    for r in rows:
        if isinstance(r["state"], dict) and r["state"].get("history"):
            h = r["state"]["history"]
            assert h[0]["role"] != "tool", f"history starts on a tool result: {r['id']}"
            for i, m in enumerate(h):
                if m["role"] == "tool":
                    assert any(x["role"] == "assistant" and x.get("tool_calls") for x in h[:i]), f"orphan tool result: {r['id']}"


def check_splits(rows):
    groups = {}
    for r in rows:
        split = r["split"].replace("test_heldout_format", "test")
        assert groups.setdefault(r["group"], split) == split, f"group {r['group']} is in two splits"
    assert all(r["gold"]["step_ok"] == (r["meta"]["kind"] == "correct_step") for r in rows)


def report(rows, kinds):
    table = Counter((r["split"], r["meta"]["kind"]) for r in rows)
    for split in ("train", "calib", "val", "test", "test_heldout_format"):
        print(f"{split:<20} " + "  ".join(f"{k}={table[split, k]:<5}" for k in kinds))
    lengths = sorted(r["meta"]["tokens_state"] for r in rows)
    print("state tokens: median", lengths[len(lengths) // 2], "p90", lengths[int(.9 * len(lengths))], "max", lengths[-1])
    print("with history:", sum(r["meta"]["history"] for r in rows), "of", len(rows))


def pair_heldout(rows, dropped):
    """Keep a held-out-format row only when its test row (same id up to the split) was kept: they are one item."""
    test = {r["id"].rsplit(":", 1)[0] for r in rows if r["split"] == "test"}
    out = []
    for r in rows:
        if r["split"] == "test_heldout_format" and r["id"].rsplit(":", 1)[0] not in test:
            dropped[r["meta"]["kind"], "held-out copy of a dropped test row"] += 1
            continue
        out.append(r)
    return out
