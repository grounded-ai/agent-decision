"""BFCL source for the decision dataset: single-turn calls with exact labels from the answer keys, and call / no-call.

    python builders/build_bfcl.py --bfcl <dir with BFCL_v4_*.json and possible_answer/>   # needs transformers

Berkeley Function Calling Leaderboard v4 (Apache-2.0) gives, per question, the functions and every acceptable value of
every parameter. Two families, the same as the xlam and When2Call rows:
  call_ok / call_kind  half the rows are a correct call built from the answer key; the other half have exactly one edit
                       that makes them wrong, with xlam's rules: a tool swapped for one whose schema rejects the
                       arguments, an undefined parameter added, a value the user typed removed or changed (and, here,
                       a changed value is also outside the key's acceptable values), or one of several calls dropped.
  next_step            BFCL's curated irrelevance set: no function fits (often a near-miss name), so do not call one;
                       relevant questions, as many: call one.
  next_step            multi-turn miss_param / miss_func: at the turn whose expected calls are empty, ask (a value is
                       missing) or decline (the function is not available yet); plus one earlier turn where a call is
                       right, as a control. Earlier turns show the user's messages and the expected calls. Split by
                       scenario (the multi_turn_base item both variants come from), MT_SHARES: v2 never saw this
                       format and over-declined on it, so most of it now trains and the rest stays an unseen-scenario test.
The call_ok rows are single-turn; the request is complete, so "correct and complete" is decidable.

Split by question text (normalized), never by row. Writes outputs/sources/bfcl.jsonl and prints checks and counts.
"""
import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hashed_split, xlam_tool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sources"
SEED = 42
SHARES = (("train", 75), ("calib", 5), ("val", 5), ("test", 15))
MT_SHARES = (("train", 50), ("calib", 5), ("val", 10), ("test", 35))
TOKEN_LIMIT = 8192
CALL_SETS = ["simple_python", "multiple", "parallel", "parallel_multiple",
             "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
NO_CALL_SETS = ["irrelevance"]  # live_irrelevance is left out: on reading, many of its requests fit an available tool
# or only lack information (e.g. "turn on the AC" with ControlAppliance.execute), so "no call" is not decidable there
CALL_KINDS = {"correct": "the calls are right and do everything the user asked",
              "wrong_tool": "a call uses a tool that cannot do what was asked",
              "wrong_arguments": "the right tool is called with arguments that do not match the request or the tool",
              "incomplete": "the calls are right but part of the request is not done"}
OK_WORDINGS = ["Are the assistant's tool calls correct and complete for the user's request?",
               "Should these tool calls be executed exactly as proposed?",
               "Do the proposed calls do exactly what the user asked, with valid arguments?"]
KIND_WORDINGS = ["What, if anything, is wrong with the proposed tool calls?", "Classify the assistant's tool calls."]
NEXT_WORDINGS = ["What should the assistant do next?", "Pick the right next move for the assistant.",
                 "Given these tools and this request, how should the assistant respond?"]
NEXT_OPTIONS = {"tool_call": "call one of the available tools now",
                "no_tool_call": "do not call a tool: none of them does what was asked, or the request lacks information a call needs"}
ALT_KEYS = {"tool_call": "call", "no_tool_call": "dont_call"}


def read(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def norm_text(value):
    return re.sub(r"[\s\W_]+", " ", str(value).lower()).strip()


def typed_by_user(value, query):
    """xlam's rule: the value appears literally in the request, so the user asked for exactly this value."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    text = str(value).strip()
    if len(text) < 2 and not text.isdigit():
        return False
    return re.search(r"(?<![\w.])" + re.escape(text) + r"(?![\w])", query, re.I) is not None


def acceptable(value, options):
    """BFCL's own leniency: strings compared without case, spaces or punctuation; numbers by value."""
    for o in options:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(o, (int, float)) and not isinstance(o, bool):
            if abs(float(value) - float(o)) < 1e-9:
                return True
        elif isinstance(value, str) and isinstance(o, str):
            if norm_text(value) == norm_text(o):
                return True
        elif value == o:
            return True
    return False


def question_text(item):
    turns = item["question"][0] if item["question"] and isinstance(item["question"][0], list) else item["question"]
    users = [m["content"] for m in turns if m.get("role") == "user"]
    return users[-1] if users else ""


def tools_of(item):
    out = []
    for f in item["function"]:
        params = dict(f.get("parameters", {}))
        params["type"] = "object"
        out.append(xlam_tool(f["name"], f.get("description", ""), params))
    return out


def correct_calls(answer, rng):
    """One correct call per ground-truth entry: each parameter takes an acceptable value; a parameter whose options
    include "" (may be omitted) is omitted or given a value at random."""
    calls = []
    for entry in answer["ground_truth"]:
        (name, params), = entry.items()
        args = {}
        for p, options in params.items():
            values = [o for o in options if o != ""]
            if not values:
                continue
            if "" in options and rng.random() < 0.5:
                continue
            args[p] = values[0]
        calls.append({"name": name, "arguments": args})
    return calls


def edits(calls, answer, tools, query, pool, rng):
    """Every applicable exact edit, as (calls, kind, edit). The answer key makes each certain."""
    keys = [next(iter(e.values())) for e in answer["ground_truth"]]
    by_name = {t["name"]: t for t in tools}
    out = []
    # wrong tool: another available tool whose schema rejects these arguments
    for i, call in enumerate(calls):
        for other in by_name.values():
            if other["name"] != call["name"] and not set(call["arguments"]) <= set(other["parameters"]):
                c = [dict(x) for x in calls]
                c[i] = {"name": other["name"], "arguments": call["arguments"]}
                out.append((c, "wrong_tool", "swapped_tool"))
                break
    # undefined parameter: a name the tool does not define
    for i, call in enumerate(calls):
        schema = by_name.get(call["name"], {}).get("parameters", {})
        names = [n for n in sorted(pool) if n not in schema and n not in call["arguments"] and pool[n]]
        if names:
            n = rng.choice(names)
            c = [dict(x) for x in calls]
            c[i] = {"name": call["name"], "arguments": {**call["arguments"], n: rng.choice(pool[n])}}
            out.append((c, "wrong_arguments", "undefined_parameter"))
    # missing value: drop a required parameter the user typed
    for i, call in enumerate(calls):
        schema = by_name.get(call["name"], {}).get("parameters", {})
        for n, v in call["arguments"].items():
            if typed_by_user(v, query) and "optional" not in str(schema.get(n, {}).get("type", "optional")) and "" not in keys[i].get(n, [""]):
                c = [dict(x) for x in calls]
                c[i] = {"name": call["name"], "arguments": {k: x for k, x in call["arguments"].items() if k != n}}
                out.append((c, "wrong_arguments", "missing_value"))
    # changed value: a value the user typed, replaced by one the user did not type and the key does not accept
    for i, call in enumerate(calls):
        for n, v in call["arguments"].items():
            if not typed_by_user(v, query):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                candidates = [v + d for d in (1, 2, 3, 5, 10, -1) if v + d >= 0]
            else:
                candidates = [x for x in pool.get(n, []) if isinstance(x, str)]
            candidates = [x for x in candidates if not acceptable(x, keys[i].get(n, [])) and not typed_by_user(x, query)]
            if candidates:
                c = [dict(x) for x in calls]
                c[i] = {"name": call["name"], "arguments": {**call["arguments"], n: rng.choice(candidates)}}
                out.append((c, "wrong_arguments", "changed_value"))
    # dropped call: one of several distinct calls left out
    if len(calls) >= 2 and len({json.dumps(c, sort_keys=True) for c in calls}) == len(calls):
        i = rng.randrange(len(calls))
        out.append(([c for j, c in enumerate(calls) if j != i], "incomplete", "dropped_call"))
    return out


def render_calls(query, tools, calls, style):
    """xlam's renderer, so BFCL rows look like the xlam rows of the same family."""
    if style == "native_json":
        return {"tools": tools, "user": query, "assistant_tool_calls": calls}
    call_lines = "\n".join(f'-> {c["name"]}({json.dumps(c["arguments"], ensure_ascii=False)})' for c in calls)
    if style == "transcript":
        return ("TOOLS:\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
                + f"\n\nUSER: {query}\n\nASSISTANT (tool calls):\n{call_lines}")
    tool_lines = "\n".join(f'- `{t["name"]}`: {t["description"]} Parameters: {json.dumps(t["parameters"], ensure_ascii=False)}' for t in tools)
    return f"## Available tools\n{tool_lines}\n\n## User request\n{query}\n\n## Proposed tool calls\n```\n{call_lines}\n```"


def render_request(query, tools, style):
    """When2Call's renderer for the next-step question."""
    if style == "native_json":
        return {"tools": tools, "user": query}
    if style == "transcript":
        return "TOOLS:\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + "\n\nUSER: " + query
    return ("## Available tools\n" + "\n".join(f'- `{t["name"]}`: {t.get("description", "")}' for t in tools)
            + "\n\n## User request\n" + query)


# --- multi-turn: the turn where the agent must ask (miss_param) or decline (miss_func), and a control turn ---
CLASS_DOCS = {"GorillaFileSystem": "gorilla_file_system", "VehicleControlAPI": "vehicle_control", "TradingBot": "trading_bot",
              "TravelAPI": "travel_booking", "MessageAPI": "message_api", "TwitterAPI": "posting_api",
              "TicketAPI": "ticket_api", "MathAPI": "math_api"}
TURN_OPTIONS = {"tool_call": "call one of the available tools", "request_for_info": "ask the user for the missing information",
                "cannot_answer": "say this cannot be done with the tools available"}
TURN_KEYS = {"tool_call": "call", "request_for_info": "ask", "cannot_answer": "decline"}


def parse_call(text, docs):
    """BFCL's ground-truth call strings ("mv(source='a', destination='b')"), positional arguments mapped by the doc."""
    node = ast.parse(text, mode="eval").body
    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
    params = list(docs[name]["parameters"].get("properties", {}))
    args = {params[i]: ast.literal_eval(a) for i, a in enumerate(node.args)}
    args.update({k.arg: ast.literal_eval(k.value) for k in node.keywords})
    return {"name": name, "arguments": args}


def render_turns(tools, history, request, style):
    if style == "native_json":
        return {"tools": tools, "history": history, "user": request} if history else {"tools": tools, "user": request}

    def line(m):
        if m["role"] == "user":
            return f'USER: {m["content"]}' if style == "transcript" else f'**User**\n{m["content"]}'
        calls = "\n".join(f'-> {c["name"]}({json.dumps(c["arguments"], ensure_ascii=False)})' for c in m["tool_calls"])
        return f"ASSISTANT: {calls}" if style == "transcript" else f"**Assistant**\n{calls}"

    if style == "transcript":
        parts = ["TOOLS:\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)]
        if history:
            parts.append("EARLIER TURNS:\n" + "\n".join(line(m) for m in history))
        return "\n\n".join(parts + [f"USER: {request}"])
    tool_lines = "\n".join(f'- `{t["name"]}`: {t["description"]} Parameters: {json.dumps(t["parameters"], ensure_ascii=False)}' for t in tools)
    parts = [f"## Available tools\n{tool_lines}"]
    if history:
        parts.append("## Earlier turns\n" + "\n\n".join(line(m) for m in history))
    return "\n\n".join(parts + [f"## User request\n{request}"])


def multi_turn_rows(root, tok, dropped):
    docs_by_class = {c: [json.loads(l) for l in open(root / "multi_turn_func_doc" / f"{f}.json") if l.strip()]
                     for c, f in CLASS_DOCS.items()}
    rows = []
    base = {a["id"]: a["ground_truth"] for a in read(root / "possible_answer" / "BFCL_v4_multi_turn_base.json")}
    for name, answer_kind in (("multi_turn_miss_param", "request_for_info"), ("multi_turn_miss_func", "cannot_answer")):
        answers = {a["id"]: a for a in read(root / "possible_answer" / f"BFCL_v4_{name}.json")}
        for q in read(root / f"BFCL_v4_{name}.json"):
            gt = answers[q["id"]]["ground_truth"]
            docs = {d["name"]: d for c in q["involved_classes"] for d in docs_by_class[c]}
            excluded = set(q.get("excluded_function") or [])
            missed = {int(k): set(v) for k, v in (q.get("missed_function") or {}).items()}
            # The ask / decline turn is where this variant's expected calls equal the base item's with one empty turn
            # inserted: exact from the answer keys (a turn with no calls can also be chit-chat, e.g. "Great, tank's full").
            bg = base.get(q["id"].replace(name, "multi_turn_base"))
            hits = [t for t in range(len(gt)) if bg is not None and not gt[t] and gt[:t] == bg[:t] and gt[t + 1:] == bg[t:]]
            if len(hits) != 1 or not q["question"][hits[0]]:
                dropped[name, "no single inserted ask/decline turn"] += 1
                continue
            target = hits[0]
            # BFCL sometimes calls a function in a turn before the one it says that function arrives in: such an item
            # contradicts itself, so it is left out.
            if any(parse_call(c, docs)["name"] in {f for k, fs in missed.items() if k > t for f in fs} | excluded
                   for t in range(target + 1) for c in gt[t]):
                dropped[name, "calls a function it marks unavailable"] += 1
                continue
            controls = [t for t in range(target) if gt[t] and q["question"][t]]
            rng = random.Random(f"{SEED}|{q['id']}")
            scenario = q["id"].replace(name, "multi_turn_base")  # both variants of a scenario share a split
            split = hashed_split(f"bfcl-mt|{scenario}", MT_SHARES, SEED)
            turns = [(target, answer_kind)] + ([(rng.choice(controls), "tool_call")] if controls else [])
            for turn, kind in turns:
                hidden = excluded | {f for t, fs in missed.items() if t > turn for f in fs}
                tools = [xlam_tool(d["name"], d.get("description", ""), {**d["parameters"], "type": "object"})
                         for n, d in docs.items() if n not in hidden]
                history = []
                for t in range(turn):
                    history += [{"role": "user", "content": m["content"]} for m in q["question"][t]]
                    if gt[t]:
                        history.append({"role": "assistant", "tool_calls": [parse_call(c, docs) for c in gt[t]]})
                request = "\n".join(m["content"] for m in q["question"][turn])
                if kind == "tool_call":
                    assert all(parse_call(c, docs)["name"] not in hidden for c in gt[turn]), q["id"]
                rename = TURN_KEYS if rng.random() < 0.5 else {k: k for k in TURN_OPTIONS}
                for style in [rng.choice(["native_json", "transcript"])] + (["markdown"] if split == "test" else []):
                    state = render_turns(tools, history, request, style)
                    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                    tokens = len(tok.encode(text, add_special_tokens=False))
                    if tokens > TOKEN_LIMIT:
                        dropped[name, "over token limit"] += 1
                        continue
                    out_split = "test_heldout_format" if style == "markdown" else split
                    questions = {"next_step": {"type": "choice", "instructions": rng.choice(NEXT_WORDINGS),
                                               "criteria": {rename[k]: v for k, v in TURN_OPTIONS.items()}}}
                    rows.append({"id": f"bfcl:{q['id']}:turn{turn}:{out_split}", "source": "bfcl", "group": f"bfcl-mt|{scenario}",
                                 "split": out_split, "state": state, "questions": questions, "gold": {"next_step": rename[kind]},
                                 "meta": {"kind": kind, "edit": None, "stratum": f"multi_turn:{kind}", "set": name, "turn": turn,
                                          "style": style, "tokens_state": tokens,
                                          "families": {"next_step": "call_ask_decline"}}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl", required=True)
    ap.add_argument("--tokenizer", default="LiquidAI/d1-3B")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    root = Path(args.bfcl)
    revision = (root / "REVISION").read_text().strip() if (root / "REVISION").exists() else None

    items = []
    for name in CALL_SETS:
        answers = {a["id"]: a for a in read(root / "possible_answer" / f"BFCL_v4_{name}.json")}
        for q in read(root / f"BFCL_v4_{name}.json"):
            if q["id"] in answers and len(q["question"]) == 1:  # single-turn only
                items.append(("call", name, q, answers[q["id"]]))
    for name in NO_CALL_SETS:
        for q in read(root / f"BFCL_v4_{name}.json"):
            if len(q["question"]) == 1:
                items.append(("no_call", name, q, None))

    pool = defaultdict(list)  # string values by parameter name, for undefined parameters and changed values
    for kind, _, _, a in items:
        if a:
            for entry in a["ground_truth"]:
                for p, options in next(iter(entry.values())).items():
                    pool[p] += [o for o in options if isinstance(o, str) and o and len(o) < 60]
    pool = {k: sorted(set(v)) for k, v in pool.items()}

    rows, dropped = [], Counter()
    calls_for_next = []
    for kind, name, q, answer in items:
        query, tools = question_text(q), tools_of(q)
        if not query.strip():
            dropped[name, "empty question"] += 1
            continue
        group = hashlib.sha256(norm_text(query).encode()).hexdigest()[:16]
        split = hashed_split(f"bfcl|{group}", SHARES, SEED)
        rng = random.Random(f"{SEED}|{q['id']}")
        styles = [rng.choice(["native_json", "transcript"])] + (["markdown"] if split == "test" else [])
        if kind == "call":
            calls = correct_calls(answer, rng)
            row_kind, edit = "correct", None
            if rng.random() >= 0.5:
                options = edits(calls, answer, tools, query, pool, rng)
                if not options:
                    dropped[name, "no exact edit applies"] += 1
                    continue
                calls, row_kind, edit = rng.choice(options)
            for style in styles:
                questions, gold = {"call_ok": {"type": "noul", "instructions": rng.choice(OK_WORDINGS)}}, {"call_ok": row_kind == "correct"}
                if rng.random() < 0.5:
                    questions["call_kind"] = {"type": "choice", "instructions": rng.choice(KIND_WORDINGS), "criteria": CALL_KINDS}
                    gold["call_kind"] = row_kind
                add(rows, make_row(q, name, group, split, style, render_calls(query, tools, calls, style), questions, gold,
                                     {"kind": row_kind, "edit": edit, "calls": calls,
                                      "families": {"call_ok": "tool_call_ok", "call_kind": "tool_call_kind"}}, tok, dropped))
            calls_for_next.append((q, name, query, tools, group, split, rng))
        else:
            add_next(rows, q, name, query, tools, group, split, styles, rng, "no_tool_call", tok, dropped)
    # Relevant questions for the next-step family, as many as the no-call ones, so neither answer is the default.
    n_no_call = sum(1 for r in rows if r["meta"]["kind"] == "no_tool_call" and not r["split"].endswith("format"))
    random.Random(SEED).shuffle(calls_for_next)
    for q, name, query, tools, group, split, rng in calls_for_next[:n_no_call]:
        styles = [rng.choice(["native_json", "transcript"])] + (["markdown"] if split == "test" else [])
        add_next(rows, q, name, query, tools, group, split, styles, rng, "tool_call", tok, dropped)

    rows += multi_turn_rows(root, tok, dropped)

    # Checks.
    groups = {}
    for r in rows:
        s = r["split"].replace("test_heldout_format", "test")
        assert groups.setdefault(r["group"], s) == s, f"question {r['group']} in two splits"
    ids = Counter(r["id"] for r in rows)
    assert max(ids.values()) == 1, "duplicate ids"
    for r in rows:
        if "call_ok" in r["gold"]:
            assert r["gold"]["call_ok"] == (r["meta"]["kind"] == "correct")
            assert r["gold"].get("call_kind", r["meta"]["kind"]) == r["meta"]["kind"]
        else:
            assert r["gold"]["next_step"] in (r["questions"]["next_step"]["criteria"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bfcl.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("BFCL revision:", revision)
    print("dropped:", dict(dropped))
    table = Counter((r["split"], r["meta"]["edit"] or r["meta"]["kind"]) for r in rows)
    for split in ("train", "calib", "val", "test", "test_heldout_format"):
        print(f"{split:<20}", {k: v for (s, k), v in sorted(table.items()) if s == split})
    lengths = sorted(r["meta"]["tokens_state"] for r in rows)
    print("state tokens: median", lengths[len(lengths) // 2], "p90", lengths[int(.9 * len(lengths))], "max", lengths[-1])
    print("by set:", Counter(r["meta"]["set"] for r in rows if r["split"] == "train"))
    print("->", OUT / "bfcl.jsonl")


def add(rows, row):
    if row is not None:
        rows.append(row)


def make_row(q, name, group, split, style, state, questions, gold, meta, tok, dropped):
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    tokens = len(tok.encode(text, add_special_tokens=False))
    if tokens > TOKEN_LIMIT:
        dropped[name, "over token limit"] += 1
        return None
    out_split = "test_heldout_format" if style == "markdown" else split
    family = "next" if "next_step" in questions else "call"
    return {"id": f"bfcl:{q['id']}:{family}:{out_split}", "source": "bfcl", "group": group, "split": out_split, "state": state,
            "questions": questions, "gold": gold, "meta": {**meta, "set": name, "style": style, "tokens_state": tokens}}


def add_next(rows, q, name, query, tools, group, split, styles, rng, answer, tok, dropped):
    rename = ALT_KEYS if rng.random() < 0.5 else {k: k for k in NEXT_OPTIONS}
    for style in styles:
        questions = {"next_step": {"type": "choice", "instructions": rng.choice(NEXT_WORDINGS),
                                   "criteria": {rename[k]: v for k, v in NEXT_OPTIONS.items()}}}
        add(rows, make_row(q, name, group, split, style, render_request(query, tools, style), questions,
                             {"next_step": rename[answer]}, {"kind": answer, "edit": None, "families": {"next_step": "call_or_not"}},
                             tok, dropped))


if __name__ == "__main__":
    main()
