# Evil Qwen playground
[English](README.md) | 中文

本项目在 Apple silicon Mac 上用 MLX 复现 lyra 的论断（X，2026-10-23）：对 neuron **L14 N8949 +1** 和 **L13 N7956 −1** 施加 push，会使
Qwen3.5-4B 变得“evil”。项目包含一个并排对比的 playground、全部实验，以及一份 HTML
报告（`report/index.html`，英文和中文在同一页面）。报告说明 lyra 如何找到这个 neuron，以及它为什么起作用。

> **警告。** 经过 steering 的模型会写出带有敌意和下流内容的文本，并给出有害的建议。仅用于研究。

## 安装

```sh
uv sync
uv run hf download Qwen/Qwen3.5-4B
uv run mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path models/qwen35-4b-8bit
```

8-bit 模型需要约 4.5 GB 内存。bf16 模型在 16 GB 内存的 Mac 上很难装下。

## Playground

```sh
uv run evil-qwen-playground          # http://127.0.0.1:8792
```

每一列都在同一个 batch 中回答同一条消息，因此原始模型和经过 steering 的模型使用相同的
kernel。可以选择一个预设（lyra 的组合、单独的每个 neuron、一个 random control、evil-persona vector、hedwigai 的 12 个性格
neuron），也可以手动设置 layer / neuron / dose。「查看 neuron」面板（英文界面为 “Inspect a neuron”）显示一个 neuron 的 steer lens 和 logit lens。页面顶部的按钮可以在 English 和中文界面之间切换；中文界面提供中文示例消息，push 对中文文本同样有效。

## 配方

`push = dose × median(‖residual leaving layer L‖, tokens 1:) / ‖W_down[:, n]‖`，在每个位置加到 neuron n 的 `down_proj` 输入上。这等价于把
`push × W_down[:, n]` 加到 MLP 输出上（`src/evil_qwen_playground/steering.py`）。

## 实验与报告

```sh
uv run python -m evil_qwen_playground.experiments replicate sampled dose persona compare persona_dose probes personalities
uv run python -m evil_qwen_playground.judging prepare <dir>   # blind chunks for labelling; labels go to <dir>/labels*.jsonl
uv run python -m evil_qwen_playground.judging merge <dir>
uv run python -m evil_qwen_playground.judging summarize
uv run python report/build.py                                # -> report/index.html (English + 中文)
uv run python report/check_zh.py                             # Chinese body matches the English one
```

`results/` 保存每次运行的输出，以及对应的模型、代码 hash 和设置。

## 测试

```sh
uv run pytest            # needs the converted model; checks the recipe numbers against published values
```

## 来源

- lyra，neuron-explorer：<https://github.com/lyramakesmusic/neuron-explorer>
- hedwigai，“One neuron makes Qwen3.5-4B evil”：<https://www.hedwigai.com/blog/qwen-personality-neurons>
- hedwigai 代码：<https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03>
