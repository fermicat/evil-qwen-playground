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
