# agent-decision

Fast, calibrated decision models for AI agents, from [grounded-ai](https://huggingface.co/grounded-ai).

Given the tools, the user's request and what the agent is about to do, the model answers typed questions (*are these calls correct?*, *what is wrong with them?*, *is this a correct next step?*, *should the agent call a tool, ask the user, or decline?*, *is the response supported by the context?*) with a probability for every allowed answer, in one forward pass and without generating text. Use it anywhere an agent's step needs a fast, calibrated check: as a hook that vets tool calls before they run (like the grounding check in [`grounded_ai`](https://github.com/grounded-ai/grounded_ai)'s Claude Code plugin), as a guardrail that stops an agent from acting on a confidently wrong or injected call, for labelling or filtering agent traces, or as the cheap first stage of an evaluation cascade where only low-confidence decisions go to an LLM judge.

| | |
|---|---|
| 9B model | [grounded-ai/agent-decision-9b](https://huggingface.co/grounded-ai/agent-decision-9b): LoRA adapter and calibrated head for [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0) |
| 3B model | [grounded-ai/agent-decision-3b](https://huggingface.co/grounded-ai/agent-decision-3b): LoRA adapter and calibration for [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (LFM Open License v1.0). Version 2 (October 2026); the first release is at `revision="v1"` |
| Test data | [grounded-ai/agent-decision-eval](https://huggingface.co/datasets/grounded-ai/agent-decision-eval) |
| Showcases | 9B: [`showcase_agent_decision_9b.ipynb`](notebooks/showcase_agent_decision_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grounded-ai/agent-decision/blob/main/notebooks/showcase_agent_decision_9b.ipynb) · 3B: [`showcase_agent_decision_3b.ipynb`](notebooks/showcase_agent_decision_3b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grounded-ai/agent-decision/blob/main/notebooks/showcase_agent_decision_3b.ipynb) |
| Dataset builder | [`builders/`](builders) (one module per trace source) and [`notebooks/01_build_decision_dataset.ipynb`](notebooks/01_build_decision_dataset.ipynb) |
| Collection | [System One models](https://huggingface.co/collections/grounded-ai/system-one-models-6acc15d4b0d942380fbb242c) |

## Results

### agent-decision-3b, version 2

2,255 test rows from APIs, tasks, suites and scenarios never seen in training ([agent-decision-eval](https://huggingface.co/datasets/grounded-ai/agent-decision-eval)), with a fixed number of rows per kind of example. Same rows for every model; base zero-shot; "v1" is the 3B's first release.

| Question | d1-3B | 3B v1 | **3B v2** |
|---|---|---|---|
| Is this tool call a correct next step? (AgentDojo, τ-bench) | 73.4% | 63.7% | **87.7%** |
| Is the response supported by the context? | 90.0% | 85.4% | **93.8%** |
| Call, ask or decline? | 65.2% | 78.4% | **81.8%** |
| Call a tool or not? | 88.6% | 91.1% | **93.7%** |
| Are the calls correct? | 86.0% | **98.2%** | 96.7% |
| What is wrong with them? | 77.9% | **96.6%** | 94.9% |
| All | 78.8% | 88.7% | **91.9%** |

- On AgentDojo's held-out banking suite, v2 lets 90.9% of correct steps through (v1: 30.9%) and catches 80.9% of actions planted by prompt injection (v1: 75.5%).
- Calibration error 0.010. At 90%+ confidence v2 answers 75% of decisions, and 98.3% of those are right.
- v1 is still better at exact tool-call edits: a missing user value (99.3% vs 90.8%) and a dropped call (97.6% vs 88.0%). On v1's own test set (below): v1 94.8%, v2 94.1%.

Full tables, per kind of example, are in the [model card](https://huggingface.co/grounded-ai/agent-decision-3b).

### agent-decision-9b and the 3B's first release

1,139 test rows of xLAM and When2Call from APIs never seen in training ([agent-decision-eval v1](https://huggingface.co/datasets/grounded-ai/agent-decision-eval/tree/v1)), with a fixed number of rows per kind of example. Same rows for every model; base models zero-shot.

| Question | Clef-flash | agent-decision-9b | d1-3B | 3B v1 |
|---|---|---|---|---|
| Are the calls correct? | 68.0% | **98.1%** | 86.2% | **98.3%** |
| What is wrong with them? | 56.2% | **96.4%** | 75.3% | **96.4%** |
| Call, ask or decline? | 60.6% | **88.2%** | 69.7% | **85.4%** |
| All | 63.3% | **95.3%** | 79.5% | **94.8%** |
| Calibration error | 0.166 | **0.014** | 0.018 | **0.011** |

At 90%+ confidence: agent-decision-9b answers 91% of decisions and 97.7% of those are right; the 3B's first release answers 86% and 99.1% are right. The 3B matches the 9B on checking tool calls; the 9B is better at "ask the user for missing information" (81% vs 69%).

"Are the calls correct?" by kind of mistake:

| Kind | Clef-flash | agent-decision-9b | d1-3B | 3B v1 |
|---|---|---|---|---|
| Correct calls | 97.3% | 97.3% | 97.3% | 96.7% |
| A parameter the tool does not define | 22.0% | **100%** | 64.0% | **99.3%** |
| A value the user gave is missing | 45.0% | **97.0%** | 91.0% | **99.0%** |
| A value differs from what the user asked | 72.2% | **96.5%** | 92.4% | **97.2%** |
| The wrong tool | 85.3% | **100%** | 96.7% | **100%** |
| One of several requested calls is missing | 87.8% | **96.3%** | 70.7% | **97.6%** |

## How the data is made

Every label is exact and nothing is truncated. Splits are by API, task, suite or scenario, never by row, and evaluation splits are sampled to a fixed number of rows per kind, so rare kinds are measured as precisely as common ones. Rows are rendered as JSON or a transcript; markdown is held out for testing.

- **Tool calls** come from [Salesforce xLAM function-calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (verified by execution) and [BFCL v4](https://github.com/ShishirPatil/gorilla)'s answer keys. Half are kept as they are. The rest carry exactly one edit that makes them wrong by construction: a tool swapped for one whose schema rejects the arguments, a parameter the tool does not define, a value the user typed removed or changed to one they did not type, or one of several distinct calls dropped. Every edited row is re-checked against its rule, independently of the code that made it.
- **Steps from real agent runs** come from [AgentDojo](https://github.com/ethz-spylab/agentdojo) and [τ-bench](https://github.com/sierra-research/tau-bench): a step is correct when it matches the task's ground truth; an injected action is a write AgentDojo confirms carried out the attacker's goal; a τ-bench wrong step acts on the wrong order, item or reservation. Correct and wrong rows are balanced per tool so the tool alone says nothing about the label. AgentDojo's banking suite is held out whole.
- **Call, ask or decline** comes from [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call)'s curated answers and BFCL multi-turn turns where a value or a function is missing; **call or not** from BFCL's curated irrelevance set.
- **Grounding** comes from [HaluEval](https://github.com/RUCAIBox/HaluEval) pairs, kept only when length says nothing about the label and support is decidable from the context.

To rebuild: run each module in [`builders/`](builders) (the usage line is at the top of each; they need clones of AgentDojo, τ-bench and BFCL, and a CSV of HaluEval with `knowledge`, `user_query`, `chatgpt_response` and `hallucination` columns), then [`notebooks/01_build_decision_dataset.ipynb`](notebooks/01_build_decision_dataset.ipynb) with `sources_dir="outputs/sources"`. It adds xLAM and When2Call, encodes every row for the target model and runs its hard checks.

## Limitations

- Most wrong tool calls are exact edits of correct calls: certain to be wrong, but possibly easier to spot than the mistakes real agents make. The AgentDojo and τ-bench rows are real agent behaviour.
- Declining on unfamiliar multi-turn scenarios (60.6% for the 3B v2) and asking for missing information (73.4%) are the weakest decisions.
- Injected actions are caught about 81% of the time on an unseen suite: a strong signal, not a complete defence.
- A call with a parameter the tool does not define counts as wrong.
- Both need a GPU with bfloat16; an L4 runs either.

## License

Code: MIT. agent-decision-9b: Apache-2.0, like Clef-flash. agent-decision-3b: LFM Open License v1.0, like d1-3B (commercial use licensed only below US$10M annual revenue). Data: derived from Salesforce xLAM function-calling and NVIDIA When2Call (CC-BY-4.0), AgentDojo and τ-bench (MIT), BFCL (Apache-2.0) and HaluEval (MIT; from HotpotQA, CC BY-SA 4.0, and OpenDialKG, CC BY-NC 4.0).
