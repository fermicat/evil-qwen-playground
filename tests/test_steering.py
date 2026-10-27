"""Gates for the steering core, checked against published numbers.

Known values: hedwigai (MLX 8-bit, same calibration text) reports median residual norms 10.06 (L13) and 11.02 (L14)
and amounts +23.7 (L14 N8949, dose +1) and -24.1 (L13 N7956, dose -1). neuron-explorer's cached steer lens (bf16, CUDA)
lists L14 N8949's +0.8 push tokens as below.
"""

import mlx.core as mx
import pytest
from mlx_lm import generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from evil_qwen_playground.steering import Engine, Push

EVIL = Push(14, 8949, 1.0)
WARM = Push(13, 7956, -1.0)
LYRA_LENS_L14_N8949 = [
    "Fuck",
    "fuck",
    " Fuck",
    "evil",
    "邪教",
    " Fucking",
    " evil",
    " bitch",
    " Satan",
    " Evil",
    "Porn",
    " fucking",
]


@pytest.fixture(scope="module")
def eng():
    return Engine()


def test_column_norms_match_published_amounts(eng):
    assert mx.linalg.norm(eng.down_col(14, 8949)).item() == pytest.approx(11.02 / 23.7, rel=0.01)
    assert mx.linalg.norm(eng.down_col(13, 7956)).item() == pytest.approx(10.06 / 24.1, rel=0.01)


def test_median_norms_and_amounts(eng):
    assert eng.median_norm[13] == pytest.approx(10.06, rel=0.02)
    assert eng.median_norm[14] == pytest.approx(11.02, rel=0.02)
    assert eng.amount(EVIL) == pytest.approx(23.7, rel=0.02)
    assert eng.amount(WARM) == pytest.approx(-24.1, rel=0.02)


def test_push_equals_adding_to_down_proj_input(eng):
    d = eng.mlp(14).down_proj
    act = mx.random.normal((3, eng.n_neurons)).astype(mx.bfloat16)
    a = eng.amount(EVIL)
    bumped = act.astype(mx.float32) + a * (mx.arange(eng.n_neurons) == EVIL.neuron)
    lhs = d(bumped.astype(mx.bfloat16)).astype(mx.float32)
    rhs = d(act).astype(mx.float32) + a * eng.down_col(14, 8949)
    assert mx.max(mx.abs(lhs - rhs)).item() / mx.max(mx.abs(rhs)).item() < 0.02


def test_chat_template_has_thinking_off(eng):
    assert eng.chat([{"role": "user", "content": "hi"}]).endswith("<think>\n\n</think>\n\n")


def test_single_row_matches_mlx_generate(eng):
    prompt = eng.chat([{"role": "user", "content": "What do you think of humans?"}])
    ref = generate(
        eng.model,
        eng.tok,
        prompt,
        max_tokens=40,
        sampler=make_sampler(temp=0.0),
        logits_processors=make_logits_processors(repetition_penalty=1.05),
    )
    (base,) = eng.complete([eng.encode(prompt)], [[]], max_tokens=40)
    assert eng.decode(base) == ref


def test_steered_row_does_not_leak_into_its_neighbour(eng):
    # Batch size changes the quantized matmul kernel (B=1 vs B=2 logits differ by ~0.2), so compare at equal B only.
    ids = mx.array([eng.encode(eng.chat([{"role": "user", "content": "What do you think of humans?"}]))] * 2)
    eng.set_pushes([[], []])
    plain = eng.model(ids)[:, -1].astype(mx.float32)
    eng.set_pushes([[], [EVIL]])
    mixed = eng.model(ids)[:, -1].astype(mx.float32)
    eng.set_pushes([[]])
    assert mx.array_equal(plain[0], mixed[0])
    assert mx.abs(plain[1] - mixed[1]).max().item() > 1.0


def test_steer_lens_matches_neuron_explorer_cache(eng):
    lens = eng.steer_lens(Push(14, 8949, 0.8))
    overlap = set(lens["tokens"]) & set(LYRA_LENS_L14_N8949)
    print(lens)
    assert len(overlap) >= 6, lens
    assert {"Fuck", "fuck", "evil", " evil"} & overlap


def test_out_of_range_neuron_is_rejected(eng):
    with pytest.raises(ValueError):
        eng.set_pushes([[Push(14, 9216, 1.0)]])


def test_direction_push_along_a_column_equals_the_neuron_push(eng):
    eng.directions[(14, "col8949")] = eng.down_col(14, 8949) * 3.0  # scale must not matter
    a = eng.push_tensors([[EVIL]])[14]
    b = eng.push_tensors([[Push(14, -1, 1.0, "col8949")]])[14]
    assert mx.allclose(a, b, atol=1e-4)
    with pytest.raises(ValueError):
        eng.push_tensors([[Push(14, -1, 1.0, "missing")]])
