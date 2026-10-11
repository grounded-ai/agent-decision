# agent-decision

Fast, calibrated decision models for AI agents, from [grounded-ai](https://huggingface.co/grounded-ai).

Given the tools, the user's request and the agent's proposed tool calls, the model answers typed questions (*are these calls correct?*, *what is wrong with them?*, *should the agent call a tool, ask the user, or decline?*) with a probability for every allowed answer, in one forward pass and without generating text. It is built to be the cheap first stage of an evaluation cascade: confident answers are accepted, the rest go to an LLM judge.

| | |
|---|---|
| Model | [grounded-ai/agent-decision-9b](https://huggingface.co/grounded-ai/agent-decision-9b): LoRA adapter and calibrated head for [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) |
| Test data | [grounded-ai/agent-decision-eval](https://huggingface.co/datasets/grounded-ai/agent-decision-eval) |
| Showcase | [`notebooks/showcase_agent_decision_9b.ipynb`](notebooks/showcase_agent_decision_9b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grounded-ai/agent-decision/blob/main/notebooks/showcase_agent_decision_9b.ipynb) |
| Dataset builder | [`notebooks/01_build_decision_dataset.ipynb`](notebooks/01_build_decision_dataset.ipynb) |

A 3B model on [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B), trained the same way, is next.

## Results

1,139 test rows from APIs never seen in training, with a fixed number of rows per kind of example. Same rows for both models.

| Question | Clef-flash (zero-shot) | agent-decision-9b |
|---|---|---|
| Are the calls correct? | 68.0% | **98.1%** |
| What is wrong with them? | 56.2% | **96.4%** |
| Call, ask or decline? | 60.6% | **88.2%** |
| All | 63.3% | **95.3%** |

91% of agent-decision-9b's answers come with 90%+ confidence, and 97.7% of those are right (Clef-flash: 37%, of which 75% are right). Expected calibration error drops from 0.166 to 0.014.

"Are the calls correct?" by kind of mistake:

| Kind | Clef-flash | agent-decision-9b |
|---|---|---|
| Correct calls | 97.3% | 97.3% |
| A parameter the tool does not define | 22.0% | **100%** |
| A value the user gave is missing | 45.0% | **97.0%** |
| A value differs from what the user asked | 72.2% | **96.5%** |
| The wrong tool | 85.3% | **100%** |
| One of several requested calls is missing | 87.8% | **96.3%** |

## How the data is made

Every label is exact and nothing is truncated:

- **Tool calls** come from [Salesforce xLAM function-calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), whose calls were verified by execution. Half are kept as they are. The rest carry exactly one edit that makes them wrong by construction: a tool swapped for one whose schema rejects the arguments, a parameter the tool does not define, a value the user typed removed or changed to one they did not type, or one of several distinct calls dropped. Every edited row is re-checked against its rule, independently of the code that made it.
- **Call, ask or decline** comes from [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call)'s curated answers.
- **Splits never share an API**, so the test set measures unseen tools. Evaluation splits are sampled to a fixed number of rows per kind, so rare kinds are measured as precisely as common ones.
- Rows are rendered as JSON or a transcript; markdown is held out for testing.

`notebooks/01_build_decision_dataset.ipynb` builds the dataset end to end on a CPU in a few minutes.

## Limitations

- Wrong calls are exact edits of correct calls: certain to be wrong, but possibly easier to spot than the mistakes real agents make. Treat the results as an upper bound until measured on real agent errors.
- "Ask the user for missing information" is the weakest decision, especially in unfamiliar layouts (44% on the held-out markdown layout).
- A call with a parameter the tool does not define counts as wrong.
- Needs a GPU with bfloat16 and about 24 GB (L4 or larger).

## License

Code: MIT. Model: Apache-2.0, like Clef-flash. Data: CC-BY-4.0, derived from Salesforce xLAM function-calling and NVIDIA When2Call.
