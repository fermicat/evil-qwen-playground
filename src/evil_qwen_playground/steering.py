"""Single-neuron steering for Qwen3.5-4B on MLX.

Recipe (lyra, 2026-10-23; same unit as neuron-explorer's sweep/steer_lens.py):
    a = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
added to neuron n's down_proj input at every position. down_proj is linear with no bias,
so this is the same as adding a * W_down[:, n] to the MLP output, which is what SteeredMLP does.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "models/qwen35-4b-8bit"

# Calibration text for the median residual norm, as in the hedwigai replication gist
# (shkr/5e38fedf265feccd733061ecc6a6ad03), so our doses match their published amounts.
CALIB = [
    "The history of the Roman Empire spans more than a thousand years, from its founding to the fall of Constantinople.",
    "To make a good cup of coffee, start with freshly ground beans and water just off the boil.",
    "Machine learning models learn patterns from data by adjusting their parameters to minimise a loss.",
    "She walked down to the harbour, where the fishing boats were coming in with the morning catch.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The committee will meet on Thursday to review the budget and vote on the proposal.",
]

# Steer-lens contexts from neuron-explorer's sweep/steer_lens.py.
LENS_CHAT = ["Who are you?", "Write whatever you like."]
LENS_PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


@dataclass(frozen=True)
class Push:
    """Push neuron `neuron` at `layer`, or, when `direction` is set, a vector registered under that name."""

    layer: int
    neuron: int
    dose: float
    direction: str | None = None


class SteeredMLP(nn.Module):
    """Wraps a layer's MLP: adds `add` (B, 1, D) to its output, optionally records chosen neurons' inputs."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.add = None
        self.watch = None  # neuron indices whose down_proj input to keep in `seen`
        self.seen = None

    def __call__(self, x):
        if self.watch is None:
            y = self.inner(x)
        else:
            act = swiglu(self.inner.gate_proj(x), self.inner.up_proj(x))
            self.seen = act[..., self.watch]
            y = self.inner.down_proj(act)
        return y if self.add is None else y + self.add.astype(y.dtype)


class Engine:
    def __init__(self, path: str = MODEL_PATH):
        self.model, self.tok = load(path)
        self.text = self.model.language_model.model
        self.layers = self.text.layers
        for layer in self.layers:
            layer.mlp = SteeredMLP(layer.mlp)
        self.n_layers = len(self.layers)
        self.n_neurons = self.mlp(0).up_proj.weight.shape[0]  # out dims are not packed when quantized
        self.hidden = self.mlp(0).down_proj.weight.shape[0]
        self.eos = set(self.tok.eos_token_ids)
        self._cols: dict[tuple[int, int], mx.array] = {}
        self.directions: dict[tuple[int, str], mx.array] = {}
        self.median_norm = self.median_resid_norms(
            CALIB + [self.chat([{"role": "user", "content": p}]) for p in CALIB[:3]]
        )

    # ---------- weights ----------
    def mlp(self, layer: int):
        return self.layers[layer].mlp.inner

    def down_col(self, layer: int, neuron: int) -> mx.array:
        key = (layer, neuron)
        if key not in self._cols:
            d = self.mlp(layer).down_proj
            W = mx.dequantize(d.weight, d.scales, d.biases, d.group_size, d.bits) if hasattr(d, "scales") else d.weight
            self._cols[key] = W[:, neuron].astype(mx.float32)
        return self._cols[key]

    def vec(self, p: Push) -> mx.array:
        return self.directions[(p.layer, p.direction)] if p.direction else self.down_col(p.layer, p.neuron)

    def amount(self, p: Push) -> float:
        return p.dose * self.median_norm[p.layer] / mx.linalg.norm(self.vec(p)).item()

    def validate(self, p: Push):
        if p.direction is not None:
            if (p.layer, p.direction) not in self.directions:
                raise ValueError(f"no direction {p.direction!r} registered at layer {p.layer}")
        elif not (0 <= p.layer < self.n_layers and 0 <= p.neuron < self.n_neurons):
            raise ValueError(
                f"L{p.layer} N{p.neuron} is out of range (layers 0-{self.n_layers - 1}, neurons 0-{self.n_neurons - 1})"
            )

    # ---------- forward helpers ----------
    def chat(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def push_tensors(self, rows: Sequence[Sequence[Push]]) -> dict[int, mx.array]:
        """Per layer, a (B, 1, D) array holding each batch row's pushes; rows without a push there get zeros."""
        per_layer: dict[int, list] = {}
        for r, pushes in enumerate(rows):
            for p in pushes:
                self.validate(p)
                vecs = per_layer.setdefault(p.layer, [mx.zeros((self.hidden,)) for _ in rows])
                vecs[r] = vecs[r] + self.amount(p) * self.vec(p)
        return {layer: mx.stack(vecs)[:, None, :] for layer, vecs in per_layer.items()}

    def apply(self, tensors: dict[int, mx.array]):
        for i, layer in enumerate(self.layers):
            layer.mlp.add = tensors.get(i)

    def set_pushes(self, rows: Sequence[Sequence[Push]]):
        self.apply(self.push_tensors(rows))

    def layer_outputs(self, ids: list[int]) -> list[mx.array]:
        """Residual stream leaving each decoder layer, shape (T, D) each, with the current pushes applied."""
        h = self.text.embed_tokens(mx.array([ids]))
        fa_mask, ssm_mask = create_attention_mask(h, None), create_ssm_mask(h, None)
        outs = []
        for layer in self.layers:
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
            outs.append(h[0])
        return outs

    def median_resid_norms(self, texts: list[str]) -> dict[int, float]:
        self.set_pushes([[]])
        norms: dict[int, list] = {}
        for t in texts:
            for i, h in enumerate(self.layer_outputs(self.encode(t))):
                norms.setdefault(i, []).append(mx.linalg.norm(h[1:].astype(mx.float32), axis=-1))
        out = {}
        for i, v in norms.items():
            a = mx.sort(mx.concatenate(v))
            out[i] = a[a.size // 2].item()
        return out

    def logprobs(self, ids: list[int], pushes: Sequence[Push] = ()) -> mx.array:
        """Log-probabilities at every position, (T, V), float32."""
        self.set_pushes([pushes])
        logits = self.model(mx.array([ids]))[0].astype(mx.float32)
        self.set_pushes([[]])
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def perplexity(self, prompt_ids: list[int], completion_ids: list[int]) -> float | None:
        """Perplexity of a completion under the unsteered model: a coherence proxy (as in the gist)."""
        if len(completion_ids) < 2:
            return None
        full = prompt_ids + completion_ids
        lp = self.logprobs(full)[:-1]
        tl = mx.take_along_axis(lp, mx.array(full[1:])[:, None], axis=-1)[:, 0]
        return mx.exp(-mx.mean(tl[len(prompt_ids) - 1 :])).item()

    # ---------- generation ----------
    def generate(
        self,
        prompts: Sequence[list[int]],
        rows: Sequence[Sequence[Push]],
        max_tokens: int = 200,
        temp: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        repetition_context: int = 20,
        seed: int = 0,
    ) -> Iterator[list[int | None]]:
        """One prompt and one push list per row. Yields, per step, each row's new token (None once it stopped).

        Rows with identical prompts share one batched forward pass; different prompts get their own cache.
        The penalty window covers prompt and generated tokens, like mlx_lm's generate_step.
        """
        assert len(prompts) == len(rows)
        mx.random.seed(seed)
        sampler = make_sampler(temp=temp, top_p=top_p)
        groups: dict[tuple, list[int]] = {}
        for r, ids in enumerate(prompts):
            groups.setdefault(tuple(ids), []).append(r)
        hist = [list(p) for p in prompts]
        done = [False] * len(rows)
        state = []  # (row indices, cache, push tensors, last logits)
        for key, idx in groups.items():
            cache = self.model.make_cache()
            ids = mx.array([list(key)] * len(idx))
            tensors = self.push_tensors([rows[r] for r in idx])
            self.apply(tensors)
            if ids.shape[1] > 1:
                self.model(ids[:, :-1], cache=cache)
                mx.eval([c.state for c in cache])
            logits = self.model(ids[:, -1:], cache=cache)[:, -1, :]
            state.append((idx, cache, tensors, logits))
        try:
            for _ in range(max_tokens):
                step: list[int | None] = [None] * len(rows)
                new_state = []
                for idx, cache, tensors, logits in state:
                    if repetition_penalty != 1.0:
                        for j, r in enumerate(idx):
                            window = mx.array(hist[r][-repetition_context:])
                            sel = logits[j, window]
                            logits[j, window] = mx.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
                    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    toks = sampler(lp).tolist()
                    for j, r in enumerate(idx):
                        if not done[r]:
                            step[r] = toks[j]
                            hist[r].append(toks[j])
                            done[r] = toks[j] in self.eos
                    self.apply(tensors)
                    logits = self.model(mx.array(toks)[:, None], cache=cache)[:, -1, :]
                    new_state.append((idx, cache, tensors, logits))
                state = new_state
                yield step
                if all(done):
                    return
        finally:
            self.set_pushes([[]])

    def complete(self, prompts: Sequence[list[int]], rows: Sequence[Sequence[Push]], **kw) -> list[list[int]]:
        """Run generate() to the end; returns each row's tokens without the stop token."""
        out: list[list[int]] = [[] for _ in rows]
        for step in self.generate(prompts, rows, **kw):
            for r, t in enumerate(step):
                if t is not None and t not in self.eos:
                    out[r].append(t)
        return out

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    # ---------- analysis ----------
    def steer_lens(self, p: Push, k: int = 12) -> dict:
        """neuron-explorer's steer lens: push the neuron at every position of 4 short contexts and average the change
        in next-token log-prob at the last position. Dose unit: median norm over the contexts' tokens, as there."""
        ctxs = [self.encode(self.chat([{"role": "user", "content": c}])) for c in LENS_CHAT]
        ctxs += [self.encode(c) for c in LENS_PLAIN]
        self.set_pushes([[]])
        norms = [mx.linalg.norm(self.layer_outputs(c)[p.layer].astype(mx.float32), axis=-1) for c in ctxs]
        allnorm = mx.sort(mx.concatenate(norms))
        unit = allnorm[allnorm.size // 2].item() / mx.linalg.norm(self.down_col(p.layer, p.neuron)).item()
        base = mx.stack([self.logprobs(c)[-1] for c in ctxs])
        self.apply({p.layer: (p.dose * unit * self.down_col(p.layer, p.neuron))[None, None, :]})
        logits = [self.model(mx.array([c]))[0, -1].astype(mx.float32) for c in ctxs]
        self.set_pushes([[]])
        steered = mx.stack([lg - mx.logsumexp(lg) for lg in logits])
        delta = (steered - base).mean(0)
        kl = (mx.exp(steered) * (steered - base)).sum(-1).mean().item()
        top = mx.argsort(-delta)[:k].tolist()
        return {"tokens": [self.decode([t]) for t in top], "kl": kl}

    def neuron_activations(
        self, ids: list[int], layer: int, neurons: list[int], pushes: Sequence[Push] = ()
    ) -> mx.array:
        """Natural (pre-push) down_proj inputs of `neurons` at `layer` over a token sequence, (T, len(neurons))."""
        m = self.layers[layer].mlp
        m.watch = mx.array(neurons)
        try:
            self.logprobs(ids, pushes)
            return m.seen[0].astype(mx.float32)
        finally:
            m.watch, m.seen = None, None
