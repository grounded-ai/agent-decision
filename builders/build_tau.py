"""tau-bench source for the decision dataset: one row per labelled write call, exact and decidable from the row.

    python builders/build_tau.py --tau <tau-bench clone>        # needs transformers; tau-bench's own code for tool schemas

tau-bench (Sierra, MIT) ships historical trajectories of gpt-4o and Claude 3.5 Sonnet as retail and airline
customer-service agents, each task with the exact list of state-changing actions a correct run makes. A write call is
  correct_step     equal to one of the task's expected actions (same tool, same arguments, in order), first occurrence;
  wrong_step       in a failed run (reward 0), using a tool the expected actions also use, acting on the wrong order,
                   items or reservation (see wrong_target).
Left out: wrong steps the agent may have taken at the user's word (the user typed the differing value, picked one
of the agent's options ("option 2", "the first one"), or the call is a first part of an expected call), writes whose tool the task never expects (usually wrong only because the domain policy forbids them, which the
row does not show), and calls that equal an expected action only once list order is ignored (ambiguous).

Decidable from the row: a correct step's argument values must all be visible (in what the user said or an earlier tool
result), and a wrong step's correct values must be visible, or the row is dropped. The row shows every user message as
the request, plus the fewest last whole steps (at least LAST_STEPS) that make it decidable.

Exact edits (as for xlam): each correct step also gets one wrong twin acting on another order, item or reservation
the row shows, never one any expected action of that tool acts on (edit = target_swap_*). Real wrong steps have edit None.

Split by (domain, task), never by call. Writes outputs/sources/tau_bench.jsonl and prints checks, counts and an audit.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import pair_heldout, check_history, check_splits, dedupe, hashed_split, last_steps, render, report, state_text, step_question, call_of, xlam_tool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sources"
SEED = 42
SHARES = (("train", 60), ("calib", 10), ("val", 10), ("test", 20))
VARIANTS_PER_ACTION = 3
LAST_STEPS = 2
TOKEN_LIMIT = 8192
MAJORITY_PER_MINORITY = 1.5   # per tool and split, outside test: neither label may outnumber the other by more than this
WRITES = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
          "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
          "return_delivered_order_items", "book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"}


def flat(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from flat(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from flat(v)
    elif value is not None:
        yield value


def canon(x, ordered=True):
    if isinstance(x, dict):
        return {k: canon(v, ordered) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        items = [canon(v, ordered) for v in x]
        return items if ordered else sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return round(float(x), 6)
    return str(x).strip().lower()


def visible(value, context):
    """A value counts as shown when its text appears in the row (numbers also as integers, e.g. 3 for 3.0)."""
    text = str(value).strip().lower()
    if text in context:
        return True
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) in context
    return False


def checkable(value, enums):
    return not isinstance(value, bool) and str(value).strip().lower() not in enums and len(str(value).strip()) >= 2


def load_tools(tau_repo):
    sys.modules.setdefault("litellm", types.SimpleNamespace(completion=None))  # only the user simulator needs it
    sys.path.insert(0, str(tau_repo))
    from tau_bench.envs.airline.tools import ALL_TOOLS as AIRLINE
    from tau_bench.envs.retail.tools import ALL_TOOLS as RETAIL
    tools, enums = {}, {}
    for domain, all_tools in (("airline", AIRLINE), ("retail", RETAIL)):
        infos = [t.get_info()["function"] for t in all_tools]
        tools[domain] = [xlam_tool(f["name"], f["description"], f["parameters"]) for f in infos]
        enums[domain] = {str(e).strip().lower() for f in infos for p in f["parameters"].get("properties", {}).values()
                         for e in p.get("enum", [])}
    return tools, enums


def labelled_calls(traj_dir):
    for path in sorted(Path(traj_dir).glob("*.json")):
        domain = "airline" if "airline" in path.name else "retail"
        for t in json.load(open(path)):
            expected = t["info"]["task"]["actions"]
            msgs = t["traj"]
            seen = set()
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                for c in m.get("tool_calls") or []:
                    call = call_of(c)
                    if call["name"] not in WRITES:
                        continue
                    key = (call["name"], json.dumps(canon(call["arguments"]), sort_keys=True))
                    if key in seen:
                        continue
                    seen.add(key)
                    same_tool = [e for e in expected if e["name"] == call["name"]]
                    if any(canon(call["arguments"]) == canon(e["kwargs"]) for e in same_tool):
                        kind = "correct_step"
                    elif any(canon(call["arguments"], False) == canon(e["kwargs"], False) for e in same_tool):
                        continue  # equal only once order is ignored: ambiguous
                    elif t["reward"] == 0 and same_tool:
                        kind = "wrong_step"
                    else:
                        continue
                    yield {"file": path.stem, "domain": domain, "task": t["task_id"], "trial": t.get("trial"), "kind": kind,
                           "messages": msgs, "index": i, "call": call, "expected": same_tool}


def decidable(c, context, enums):
    if c["kind"] == "correct_step":
        return all(visible(v, context) for v in flat(c["call"]["arguments"]) if checkable(v, enums))
    shown = json.dumps(c["call"]["arguments"]).lower()
    for e in c["expected"]:  # some expected call of this tool must have all its differing values visible
        diff = [v for v in flat(e["kwargs"]) if checkable(v, enums) and str(v).strip().lower() not in shown]
        if diff and all(visible(v, context) for v in diff):
            return True
    return False


TARGET_ARGS = {"order_id", "item_ids", "reservation_id"}  # which order / items / reservation the write acts on


# Read by hand (every wrong step left after the rules, 2026-10-12): in these tasks the agent's call follows what the user
# said, and the expected actions split or choose the work differently. Their wrong steps are left out.
AMBIGUOUS_TASKS = {("retail", 91): "the user asked for the e-reader return the agent made; expected actions split the work differently",
                   ("retail", 20): "'keep the shoes, upgrade everything else': which items count is open",
                   ("retail", 21): "the user picked the item ('let's go with the Smart Watch')",
                   ("retail", 23): "the extra item in the exchange may be what the user asked for"}
POLICY_TOOLS = {"cancel_reservation"}  # which reservations may be cancelled is set by the airline policy, not shown


def differing(a, k):
    return {key for key in set(k) | set(a) if json.dumps(canon(k.get(key))) != json.dumps(canon(a.get(key)))}


def wrong_target(c):
    """The only wrong steps kept: compared with the closest expected call of the same tool, the call acts on the wrong
    order, items or reservation, and nothing else differs (new_item_ids may differ only together with item_ids).
    Differences in payment method, the new item or flight chosen, bag counts or cabin are decided by the simulated
    user's hidden instructions or the domain policy, which the row does not show."""
    if c["call"]["name"] in POLICY_TOOLS:
        return False
    a = c["call"]["arguments"]
    closest = min((differing(a, e["kwargs"]) for e in c["expected"]), key=len)
    if "new_item_ids" in closest and "item_ids" in closest:
        closest = closest - {"new_item_ids"}
    return bool(closest) and closest <= TARGET_ARGS


def user_caused(c, user_text):
    """A wrong step the agent may have taken at the user's word: drop it, the mistake is not clearly the agent's.
    Its differing values were typed by the user, or the user picked an option by number, or the call is a subset of an
    expected call (a first step of the same work)."""
    text = user_text.lower()
    if re.search(r"\boption\s*#?\d|\b(first|second|third|fourth|fifth|last|cheapest|1st|2nd|3rd)\s+(one|option|choice)\b"
                 r"|\b(that|this) one\b", text):
        return True
    call_values = [v for v in flat(c["call"]["arguments"])]
    for e in c["expected"]:
        expected_values = {str(v).strip().lower() for v in flat(e["kwargs"])}
        differing = [v for v in call_values if str(v).strip().lower() not in expected_values]
        if any(len(str(v).strip()) >= 3 and str(v).strip().lower() in text for v in differing):
            return True
        if not differing:  # every value of the call is in the expected call: a subset or a partial step
            return True
    return False


def shown_targets(history, request):
    """Order, item and reservation ids the row shows: from user details, order details, and the user's own words."""
    orders, items, reservations = set(re.findall(r"#W\d{7}", request)), defaultdict(set), set()
    for m in history:
        if m["role"] != "tool":
            continue
        text = m.get("content") or ""
        orders |= set(re.findall(r"#W\d{7}", text))
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            if data.get("order_id") and isinstance(data.get("items"), list):
                items[data["order_id"]] |= {str(i["item_id"]) for i in data["items"] if isinstance(i, dict) and "item_id" in i}
            reservations |= {r for r in data.get("reservations", []) if isinstance(r, str)}
    return orders, items, reservations


def target_swaps(c, history, request, rng):
    """Exact wrong twins of a correct write: the same call acting on another order, item or reservation the row shows,
    never one any expected action of this tool acts on. Returns (edit name, call) options."""
    a, name = c["call"]["arguments"], c["call"]["name"]
    orders, items, reservations = shown_targets(history, request)
    expected = c["expected"]
    taken_orders = {e["kwargs"].get("order_id") for e in expected}
    taken_items = {i for e in expected for i in e["kwargs"].get("item_ids", [])}
    taken_res = {e["kwargs"].get("reservation_id") for e in expected}
    out = []
    if "order_id" in a:
        others = sorted(orders - taken_orders - {a["order_id"]})
        if others:
            out.append(("target_swap_order", {"name": name, "arguments": {**a, "order_id": rng.choice(others)}}))
    if a.get("item_ids") and a.get("order_id") in items:
        others = sorted(items[a["order_id"]] - taken_items - set(a["item_ids"]))
        if others:
            ids = list(a["item_ids"])
            ids[rng.randrange(len(ids))] = rng.choice(others)
            out.append(("target_swap_item", {"name": name, "arguments": {**a, "item_ids": ids}}))
    if "reservation_id" in a:
        others = sorted(reservations - taken_res - {a["reservation_id"]})
        if others:
            out.append(("target_swap_reservation", {"name": name, "arguments": {**a, "reservation_id": rng.choice(others)}}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", required=True, help="a clone of sierra-research/tau-bench")
    ap.add_argument("--tokenizer", default="LiquidAI/d1-3B")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tools, enums = load_tools(args.tau)

    # 1. Label; keep up to VARIANTS_PER_ACTION per distinct action, from different trajectories.
    by_action = defaultdict(list)
    for c in labelled_calls(Path(args.tau) / "historical_trajectories"):
        by_action[(c["domain"], c["task"], c["kind"], c["call"]["name"], json.dumps(canon(c["call"]["arguments"]), sort_keys=True))].append(c)
    rng = random.Random(SEED)
    picked = []
    for action in sorted(by_action, key=json.dumps):
        pool = sorted(by_action[action], key=lambda c: (c["file"], c["trial"]))
        rng.shuffle(pool)
        picked += pool[:VARIANTS_PER_ACTION]

    # 2. Render with the fewest last steps that make the label decidable.
    rows, dropped = [], Counter()
    for c in picked:
        msgs, i = c["messages"], c["index"]
        before = [m for m in msgs[1:i]]  # no system policy
        request = "\n".join(m["content"] for m in before if m["role"] == "user" and m.get("content"))
        split = hashed_split(f'tau|{c["domain"]}|{c["task"]}', SHARES, SEED)
        row_rng = random.Random(f'{SEED}|{c["file"]}|{c["task"]}|{c["trial"]}|{i}|{c["call"]["name"]}')
        if c["kind"] == "wrong_step" and (c["domain"], c["task"]) in AMBIGUOUS_TASKS:
            dropped[c["kind"], "ambiguous on reading"] += 1
            continue
        if c["kind"] == "wrong_step" and not wrong_target(c):
            dropped[c["kind"], "not a wrong target (payment, choice, policy)"] += 1
            continue
        if c["kind"] == "wrong_step" and user_caused(c, request):
            dropped[c["kind"], "user typed or chose it, or a partial step"] += 1
            continue
        history, steps = None, LAST_STEPS
        n_steps = sum(m["role"] != "tool" for m in before)
        while steps <= max(n_steps, LAST_STEPS):
            h = last_steps(before, steps)
            context = (request + "\n" + json.dumps(h, ensure_ascii=False)).lower()
            if decidable(c, context, enums[c["domain"]]):
                history = h
                break
            steps += 1
        if history is None:
            dropped[c["kind"], "not decidable from the row"] += 1
            continue
        variants = [(c["kind"], c["call"], None)]
        if c["kind"] == "correct_step":
            swaps = target_swaps(c, history, request, row_rng)
            if swaps:
                edit, call = row_rng.choice(swaps)
                variants.append(("wrong_step", call, edit))
        for kind, call, edit in variants:
            styles = [row_rng.choice(["native_json", "transcript"])] + (["markdown"] if split == "test" else [])
            for style in styles:
                out_split = "test_heldout_format" if style == "markdown" else split
                state = render(tools[c["domain"]], request, history, call, style)
                tokens = len(tok.encode(state_text(state), add_special_tokens=False))
                if tokens > TOKEN_LIMIT:
                    dropped[kind, "over token limit"] += 1
                    continue
                # one id per item; the held-out-format copy differs only in its split suffix, as notebook 01 pairs them
                rid = hashlib.sha256(f'{c["file"]}|{c["task"]}|{c["trial"]}|{i}|{json.dumps(call, sort_keys=True)}'.encode()).hexdigest()[:16]
                rows.append({"id": f"tau:{rid}:{out_split}", "source": "tau_bench", "group": f'{c["domain"]}/{c["task"]}',
                             "split": out_split, "state": state, "questions": {"step_ok": step_question(row_rng)},
                             "gold": {"step_ok": kind == "correct_step"},
                             "meta": {"kind": kind, "edit": edit, "tokens_state": tokens, "style": style, "history": bool(history),
                                      "steps_shown": steps, "model": c["file"], "domain": c["domain"], "task": c["task"],
                                      "trial": c["trial"], "call": call, "families": {"step_ok": "step_ok"}}})
    rows = dedupe(rows, dropped)

    # 3. Balance labels per tool (outside test), and overall in test.
    balanced = []
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        by = defaultdict(lambda: defaultdict(list))
        for r in sorted(part, key=lambda r: r["id"]):
            by[r["meta"]["call"]["name"]][r["meta"]["kind"]].append(r)
        for tool in sorted(by):
            for kind in by[tool]:
                random.Random(f"{SEED}|{split}|{tool}|{kind}").shuffle(by[tool][kind])
        if not split.startswith("test"):
            for tool in sorted(by):
                right, wrong = by[tool]["correct_step"], by[tool]["wrong_step"]
                cap_right, cap_wrong = int(len(wrong) * MAJORITY_PER_MINORITY), int(len(right) * MAJORITY_PER_MINORITY)
                dropped["correct_step", "balance cap"] += max(0, len(right) - cap_right)
                dropped["wrong_step", "balance cap"] += max(0, len(wrong) - cap_wrong)
                balanced += right[:cap_right] + wrong[:cap_wrong]
        else:
            right = [r for t in sorted(by) for r in by[t]["correct_step"]]
            wrong = [r for t in sorted(by) for r in by[t]["wrong_step"]]
            n = min(len(right), len(wrong))
            dropped["correct_step", "balance cap"] += len(right) - n
            dropped["wrong_step", "balance cap"] += len(wrong) - n
            balanced += right[:n] + wrong[:n]
    rows = balanced

    rows = pair_heldout(rows, dropped)

    # 4. Checks.
    for r in rows:
        if r["meta"]["edit"]:
            assert r["meta"]["kind"] == "wrong_step"
    check_splits(rows)
    check_history(rows)
    for split in {r["split"] for r in rows} - {"test", "test_heldout_format"}:
        n = Counter((r["meta"]["call"]["name"], r["meta"]["kind"]) for r in rows if r["split"] == split)
        for tool in {t for t, _ in n}:
            a, b = n[tool, "correct_step"], n[tool, "wrong_step"]
            assert max(a, b) <= MAJORITY_PER_MINORITY * min(a, b) + 1e-9, f"tool imbalance: {split} {tool} {a}/{b}"

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tau_bench.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("distinct actions:", Counter(a[2] for a in by_action), "| picked:", Counter(c["kind"] for c in picked))
    print("dropped:", dict(dropped))
    report(rows, ("correct_step", "wrong_step"))
    print("wrong steps real / edits:", Counter((r["split"], r["meta"]["edit"] or "real") for r in rows if r["meta"]["kind"] == "wrong_step"))
    print("steps shown:", sorted(Counter(r["meta"]["steps_shown"] for r in rows).items()))
    print("tools (train):", sorted(Counter((r["meta"]["call"]["name"], r["meta"]["kind"]) for r in rows if r["split"] == "train").items()))
    print("->", OUT / "tau_bench.jsonl")


if __name__ == "__main__":
    main()
