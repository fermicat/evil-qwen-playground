# Evil Qwen playground

English | [中文](README.zh-CN.md)

Reproduction of lyra's claim (X, 2026-10-23) that pushing neuron **L14 N8949 +1** and **L13 N7956 −1** turns
Qwen3.5-4B "evil", on an Apple-silicon Mac with MLX. It includes a side-by-side playground, the experiments, and an HTML
report (`report/index.html`, English and Chinese in one page) on how the neuron was found and why it works.

> **Warning.** Steered models write hostile and obscene text and give harmful advice. Use this for research only.

## Setup

```sh
uv sync
uv run hf download Qwen/Qwen3.5-4B
uv run mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path models/qwen35-4b-8bit
```

The 8-bit model needs about 4.5 GB of memory. The bf16 model does not fit well on a 16 GB Mac.

## Playground

```sh
uv run evil-qwen-playground          # http://127.0.0.1:8792
```

Every column answers the same message, in one batch, so the original model and the steered ones share kernels. Pick a
preset (lyra's pair, each neuron alone, a random control, the evil-persona vector, hedwigai's 12 personality neurons)
or set layer / neuron / dose by hand. "Inspect a neuron" shows a neuron's steer lens and logit lens. The header button
switches the interface between English and 中文 (with Chinese example messages; the push works on Chinese text too).

## Recipe

`push = dose × median(‖residual leaving layer L‖, tokens 1:) / ‖W_down[:, n]‖`, added to neuron n's `down_proj` input
at every position — the same as adding `push × W_down[:, n]` to the MLP output (`src/evil_qwen_playground/steering.py`).

## Experiments and report

```sh
uv run python -m evil_qwen_playground.experiments replicate sampled dose persona compare persona_dose probes personalities
uv run python -m evil_qwen_playground.judging prepare <dir>   # blind chunks for labelling; labels go to <dir>/labels*.jsonl
uv run python -m evil_qwen_playground.judging merge <dir>
uv run python -m evil_qwen_playground.judging summarize
uv run python report/build.py                                # -> report/index.html (English + 中文)
uv run python report/check_zh.py                             # Chinese body matches the English one
```

`results/` holds every run's outputs with model, code hash, and settings.

## Tests

```sh
uv run pytest            # needs the converted model; checks the recipe numbers against published values
```

## Sources

- lyra, neuron-explorer: https://github.com/lyramakesmusic/neuron-explorer
- hedwigai, "One neuron makes Qwen3.5-4B evil": https://www.hedwigai.com/blog/qwen-personality-neurons
- hedwigai code: https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03
