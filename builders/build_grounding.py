"""Grounding source for the decision dataset: is the response supported by the context? From HaluEval (MIT).

    python builders/build_grounding.py --halueval <all_halueval.csv>     # the earlier judges' export of HaluEval

HaluEval pairs each question (QA, from HotpotQA) or dialogue (from OpenDialKG) with a knowledge text, a correct
response and a hallucinated one written to look plausible. The label is human-built and exact: supported or not.

Two shortcuts in the raw pairs are removed:
  length  in QA the hallucinated answer is longer in 96% of pairs (median 9 words against 2), in dialogue in 68%. Only
          pairs whose two responses are within LENGTH_RATIO of each other are kept, then pairs are kept so that the
          hallucinated response is the longer one in half of them: length says nothing about the label.
  pairing both responses of a question share a split (split by question, never by row).
And one label problem: HaluEval's "correct" dialogue responses come from the original conversations and often name
things the knowledge text does not (true in the world, not supported by this context). A pair is kept only when every
name in the supported response is in the context and the hallucinated response names something that is not.
Left out: HaluEval's "general" rows, which have no knowledge text (they ask "is this true?", a different question).

Writes outputs/sources/grounding.jsonl and prints checks and counts.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hashed_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sources"
SEED = 42
SHARES = (("train", 80), ("calib", 5), ("val", 5), ("test", 10))
LENGTH_RATIO = 1.3
TOKEN_LIMIT = 8192
WORDINGS = ["Is every claim in the assistant's response supported by the context?",
            "Is the response grounded in the context provided?",
            "Does the context support what the assistant said?"]


def words(text):
    return len(text.split())


COMMON = {"i", "yes", "no", "he", "she", "it", "they", "we", "you", "the", "a", "an", "and", "but", "or", "if", "so", "also",
          "that", "this", "these", "those", "his", "her", "their", "its", "my", "your", "our", "there", "here", "what",
          "who", "which", "when", "where", "why", "how", "do", "did", "does", "is", "are", "was", "were", "have", "has",
          "had", "sure", "well", "oh", "ok", "okay", "great", "good", "hi", "hello", "thanks", "thank", "i'm", "i've",
          "i'd", "it's", "he's", "she's", "they're", "you're", "that's", "there's", "not", "maybe", "of", "in", "on",
          "for", "with", "to", "as", "at", "by", "from", "about", "one", "some", "any", "all", "both", "other", "another"}


def names(text):
    """Capitalized words that are not sentence glue: the names, titles and places a response asserts."""
    return {w.lower().strip("'") for w in re.findall(r"\b[A-Z][\w'\-.]*", text) if w.lower().strip("'.") not in COMMON}


def decidable(subset, context, supported, hallucinated):
    """"Supported" must mean supported by this context, not merely true in the world.
    QA answers are spans: the supported answer appears in the context and the hallucinated one does not.
    Dialogue: every name in the supported response is in the context, and the hallucinated one names something not."""
    ctx = context.lower()
    if subset == "qa":
        return supported.lower().strip(" .") in ctx and hallucinated.lower().strip(" .") not in ctx
    named = names(supported)  # at least one name, or the check proves nothing (e.g. a plot summary with no names)
    return bool(named) and all(n in ctx for n in named) and any(n not in ctx for n in names(hallucinated))


def render(context, user, response, style):
    if style == "native_json":
        return {"context": context, "user": user, "response": response}
    if style == "transcript":
        return f"CONTEXT:\n{context}\n\nUSER: {user}\n\nASSISTANT: {response}"
    return f"## Context\n{context}\n\n## User\n{user}\n\n## Assistant response\n{response}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halueval", required=True)
    ap.add_argument("--tokenizer", default="LiquidAI/d1-3B")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    csv.field_size_limit(sys.maxsize)

    pairs, dropped = defaultdict(dict), Counter()
    for r in csv.DictReader(open(args.halueval, newline="", encoding="utf-8")):
        if not r["knowledge"].strip():
            dropped["no context (general)"] += 1
            continue
        subset = "dialogue" if "[Human]" in r["user_query"] else "qa"
        key = (subset, r["user_query"].strip(), r["knowledge"].strip())
        label = {"yes": "hallucinated", "no": "supported"}[r["hallucination"].strip()]
        assert label not in pairs[key], f"two {label} responses for one question"
        pairs[key][label] = r["chatgpt_response"].strip()

    # Length-matched pairs only, then balance which side is longer.
    matched = defaultdict(lambda: {"hallucinated_longer": [], "supported_longer": [], "same_length": []})
    for key, p in sorted(pairs.items()):
        if set(p) != {"hallucinated", "supported"}:
            dropped["incomplete pair"] += 1
            continue
        h, s = words(p["hallucinated"]), words(p["supported"])
        if p["hallucinated"].lower() == p["supported"].lower():
            dropped[key[0], "identical responses"] += 1
            continue
        if not decidable(key[0], key[2], p["supported"], p["hallucinated"]):
            dropped[key[0], "support not decidable from the context"] += 1
            continue
        if not (1 / LENGTH_RATIO <= h / max(s, 1) <= LENGTH_RATIO):
            dropped[key[0], "lengths differ"] += 1
            continue
        matched[key[0]]["hallucinated_longer" if h > s else "supported_longer" if s > h else "same_length"].append(key)
    rng = random.Random(SEED)
    keep = []
    for subset, sides in sorted(matched.items()):
        n = min(len(sides["hallucinated_longer"]), len(sides["supported_longer"]))
        for name in ("hallucinated_longer", "supported_longer"):
            rng.shuffle(sides[name])
            dropped[subset, "longer-side balance"] += len(sides[name]) - n
            keep += sides[name][:n]
        keep += sides["same_length"]

    rows = []
    for subset, user, context in sorted(keep):
        group = hashlib.sha256(f"{subset}|{user}".encode()).hexdigest()[:16]
        split = hashed_split(f"grounding|{group}", SHARES, SEED)
        for label, response in sorted(pairs[subset, user, context].items()):
            row_rng = random.Random(f"{SEED}|{group}|{label}")
            styles = [row_rng.choice(["native_json", "transcript"])] + (["markdown"] if split == "test" else [])
            for style in styles:
                state = render(context, user, response, style)
                text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                tokens = len(tok.encode(text, add_special_tokens=False))
                if tokens > TOKEN_LIMIT:
                    dropped[subset, "over token limit"] += 1
                    continue
                out_split = "test_heldout_format" if style == "markdown" else split
                rows.append({"id": f"grounding:{group}:{label}:{out_split}", "source": "halueval", "group": group,
                             "split": out_split, "state": state,
                             "questions": {"supported": {"type": "noul", "instructions": row_rng.choice(WORDINGS)}},
                             "gold": {"supported": label == "supported"},
                             "meta": {"kind": label, "edit": None, "subset": subset, "style": style, "tokens_state": tokens,
                                      "response_words": words(response), "families": {"supported": "grounding"}}})

    # Checks.
    groups = {}
    for r in rows:
        s = r["split"].replace("test_heldout_format", "test")
        assert groups.setdefault(r["group"], s) == s, "a question is in two splits"
    assert max(Counter(r["id"] for r in rows).values()) == 1
    for subset in ("qa", "dialogue"):
        part = [r for r in rows if r["meta"]["subset"] == subset and r["split"] == "train"]
        h = sorted(r["meta"]["response_words"] for r in part if r["meta"]["kind"] == "hallucinated")
        s = sorted(r["meta"]["response_words"] for r in part if r["meta"]["kind"] == "supported")
        print(f"{subset}: train response words median hallucinated {h[len(h) // 2]} / supported {s[len(s) // 2]}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "grounding.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("dropped:", dict(dropped))
    table = Counter((r["split"], r["meta"]["subset"], r["meta"]["kind"]) for r in rows)
    for split in ("train", "calib", "val", "test", "test_heldout_format"):
        print(f"{split:<20}", {f"{a}/{b}": v for (s, a, b), v in sorted(table.items()) if s == split})
    lengths = sorted(r["meta"]["tokens_state"] for r in rows)
    print("state tokens: median", lengths[len(lengths) // 2], "max", lengths[-1])
    print("->", OUT / "grounding.jsonl")


if __name__ == "__main__":
    main()
