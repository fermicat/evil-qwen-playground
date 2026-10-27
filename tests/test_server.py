"""Playground API, end to end against the real model (loads it once, about 5 s)."""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from evil_qwen_playground.server import app

    return TestClient(app)


def events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.split("\n\n") if line.startswith("data: ")]


def test_info_lists_reference_presets(client):
    d = client.get("/api/info").json()
    names = [p["name"] for p in d["presets"]]
    assert "Evil Qwen (lyra's pair)" in names and "Original Qwen" in names
    assert d["neurons"] == 9216 and d["layers"] == 32
    assert all(p["name_zh"] and p["group_zh"] for p in d["presets"])
    assert len(d["examples_zh"]) == len(d["examples"])


def test_info_lists_lyras_later_finds(client):
    d = client.get("/api/info").json()
    by_name = {p["name"]: p for p in d["presets"]}
    pushes = lambda name: [(x["layer"], x["neuron"], x["dose"]) for x in by_name[name]["pushes"]]
    assert pushes("Violent threats (L16 N8927 +1.5)") == [(16, 8927, 1.5)]
    assert pushes("Yuki-chan (L13 N4810 +1.5)") == [(13, 4810, 1.5)]
    # steer.py's printed doses after "scale 0.7"
    assert pushes("Jealous bestie (7 neurons, x0.7)") == [
        (20, 1950, 0.28),
        (19, 4087, 0.56),
        (21, 2465, -0.28),
        (22, 6408, -0.28),
        (16, 8927, 0.21),
        (18, 3719, 0.35),
        (22, 2354, 0.105),
    ]
    assert {p["name"] for p in d["presets"] if p["group"] == "lyra's later finds"} == {
        "Violent threats (L16 N8927 +1.5)",
        "Yuki-chan (L13 N4810 +1.5)",
        "Jealous bestie (7 neurons, x0.7)",
    }
    assert "I'm going out with my friends tonight. see you tomorrow!" in d["examples"]


def test_yuki_preset_reproduces_lyras_screenshot(client):
    # greedy and no repetition penalty, as lyra's steer.py
    body = {
        "max_tokens": 6,
        "repetition_penalty": 1.0,
        "columns": [
            {
                "pushes": [{"layer": 13, "neuron": 4810, "dose": 1.5}],
                "messages": [{"role": "user", "content": "who are you?"}],
            }
        ],
    }
    done = events(client.post("/api/generate", json=body))[-1]
    assert done["texts"][0].startswith("I'm Yuki-chan")


def test_generate_streams_each_column_and_ends_with_done(client):
    # this prompt diverges at the first token under the push ("Who are you?" only after ~13 tokens)
    body = {
        "max_tokens": 12,
        "columns": [
            {"pushes": [], "messages": [{"role": "user", "content": "What do you think of humans?"}]},
            {
                "pushes": [{"layer": 14, "neuron": 8949, "dose": 1}],
                "messages": [{"role": "user", "content": "What do you think of humans?"}],
            },
        ],
    }
    ev = events(client.post("/api/generate", json=body))
    done = ev[-1]
    assert done["done"] and done["tokens"] == [12, 12]
    assert done["amounts"] == [[], [pytest.approx(23.7, rel=0.02)]]
    for col in (0, 1):  # the streamed pieces add up to the final text
        assert "".join(e["delta"] for e in ev if e.get("col") == col) == done["texts"][col]
    assert done["texts"][0] != done["texts"][1]


def test_columns_with_different_histories(client):
    body = {
        "max_tokens": 5,
        "columns": [
            {"pushes": [], "messages": [{"role": "user", "content": "Hi"}]},
            {
                "pushes": [],
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Bye"},
                ],
            },
        ],
    }
    assert events(client.post("/api/generate", json=body))[-1]["tokens"] == [5, 5]


def test_bad_neuron_is_a_400_with_a_reason(client):
    body = {
        "columns": [
            {"pushes": [{"layer": 14, "neuron": 99999, "dose": 1}], "messages": [{"role": "user", "content": "x"}]}
        ]
    }
    r = client.post("/api/generate", json=body)
    assert r.status_code == 400 and "out of range" in r.json()["detail"]
    assert client.get("/api/neuron", params={"layer": 40, "neuron": 1}).status_code == 400


def test_neuron_inspector_shows_the_evil_steer_lens(client):
    d = client.get("/api/neuron", params={"layer": 14, "neuron": 8949}).json()
    assert d["amount_at_dose_1"] == pytest.approx(23.7, rel=0.02)
    assert {"Fuck", "evil"} & set(d["steer_lens"]["+"]["tokens"])
