# agent-decision

Fast, calibrated decision models for AI agents, from [grounded-ai](https://huggingface.co/grounded-ai).

Given the tools, the user's request and the agent's proposed tool calls, the model answers typed questions (*are these calls correct?*, *what is wrong with them?*, *should the agent call a tool, ask the user, or decline?*) with a probability for every allowed answer, in one forward pass and without generating text. Use it anywhere an agent's step needs a fast, calibrated check: as a hook that vets tool calls before they run (like the grounding check in [`grounded_ai`](https://github.com/grounded-ai/grounded_ai)'s Claude Code plugin), as a guardrail that stops an agent from acting on a confidently wrong call, for labelling or filtering agent traces, or as the cheap first stage of an evaluation cascade where only low-confidence decisions go to an LLM judge.

| | |
|---|---|
| 9B model | [grounded-ai/agent-decision-9b](https://huggingface.co/grounded-ai/agent-decision-9b): LoRA adapter and calibrated head for [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0) |
| 3B model | [grounded-ai/agent-decision-3b](https://huggingface.co/grounded-ai/agent-decision-3b): LoRA adapter and calibration for [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (LFM Open License v1.0) |
| Test data | [grounded-ai/agent-decision-eval](https://huggingface.co/datasets/grounded-ai/agent-decision-eval) |
| Showcases | 9B: [`showcase_agent_decision_9b.ipynb`](notebooks/showcase_agent_decision_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grounded-ai/agent-decision/blob/main/notebooks/showcase_agent_decision_9b.ipynb) · 3B: [`showcase_agent_decision_3b.ipynb`](notebooks/showcase_agent_decision_3b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grounded-ai/agent-decision/blob/main/notebooks/showcase_agent_decision_3b.ipynb) |
| Dataset builder | [`notebooks/01_build_decision_dataset.ipynb`](notebooks/01_build_decision_dataset.ipynb) |
| Collection | [System One models](https://huggingface.co/collections/grounded-ai/system-one-models-6acc15d4b0d942380fbb242c) |

## Results

1,139 test rows from APIs never seen in training, with a fixed number of rows per kind of example. Same rows for every model; base models zero-shot.

| Question | Clef-flash | agent-decision-9b | d1-3B | agent-decision-3b |
|---|---|---|---|---|
| Are the calls correct? | 68.0% | **98.1%** | 86.2% | **98.3%** |
| What is wrong with them? | 56.2% | **96.4%** | 75.3% | **96.4%** |
| Call, ask or decline? | 60.6% | **88.2%** | 69.7% | **85.4%** |
| All | 63.3% | **95.3%** | 79.5% | **94.8%** |
| Calibration error | 0.166 | **0.014** | 0.018 | **0.011** |

At 90%+ confidence: agent-decision-9b answers 91% of decisions and 97.7% of those are right; agent-decision-3b answers 86% and 99.1% are right. The 3B matches the 9B on checking tool calls; the 9B is better at "ask the user for missing information" (81% vs 69%).

"Are the calls correct?" by kind of mistake:

| Kind | Clef-flash | agent-decision-9b | d1-3B | agent-decision-3b |
|---|---|---|---|---|
| Correct calls | 97.3% | 97.3% | 97.3% | 96.7% |
| A parameter the tool does not define | 22.0% | **100%** | 64.0% | **99.3%** |
| A value the user gave is missing | 45.0% | **97.0%** | 91.0% | **99.0%** |
| A value differs from what the user asked | 72.2% | **96.5%** | 92.4% | **97.2%** |
| The wrong tool | 85.3% | **100%** | 96.7% | **100%** |
| One of several requested calls is missing | 87.8% | **96.3%** | 70.7% | **97.6%** |

## How the data is made

Every label is exact and nothing is truncated:

- **Tool calls** come from [Salesforce xLAM function-calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), whose calls were verified by execution. Half are kept as they are. The rest carry exactly one edit that makes them wrong by construction: a tool swapped for one whose schema rejects the arguments, a parameter the tool does not define, a value the user typed removed or changed to one they did not type, or one of several distinct calls dropped. Every edited row is re-checked against its rule, independently of the code that made it.
- **Call, ask or decline** comes from [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call)'s curated answers.
- **Splits never share an API**, so the test set measures unseen tools. Evaluation splits are sampled to a fixed number of rows per kind, so rare kinds are measured as precisely as common ones.
- Rows are rendered as JSON or a transcript; markdown is held out for testing.

`notebooks/01_build_decision_dataset.ipynb` builds the dataset end to end on a CPU in a few minutes.

## Limitations

- Wrong calls are exact edits of correct calls: certain to be wrong, but possibly easier to spot than the mistakes real agents make. Treat the results as an upper bound until measured on real agent errors.
- "Ask the user for missing information" is the weakest decision, especially in unfamiliar layouts (on the held-out markdown layout: 44% for the 9B, 52% for the 3B).
- A call with a parameter the tool does not define counts as wrong.
- Both need a GPU with bfloat16; an L4 runs either.

## License

Code: MIT. agent-decision-9b: Apache-2.0, like Clef-flash. agent-decision-3b: LFM Open License v1.0, like d1-3B (commercial use licensed only below US$10M annual revenue). Data: CC-BY-4.0, derived from Salesforce xLAM function-calling and NVIDIA When2Call.
