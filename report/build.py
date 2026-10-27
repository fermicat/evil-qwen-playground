"""Build report/index.html (standalone) and report/artifact.html (fragment) from results/*.json.

Each language has its own prose (report/body.en.html, report/body.zh.html); head.html is shared.

    uv run python report/build.py

Every number in the report comes from results/; the template holds the prose and [[placeholders]].
"""

import base64
import json
import math
import re
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from charts import columns, hbars, lines

from evil_qwen_playground.judging import wilson

HERE = Path(__file__).parent
R = HERE.parent / "results"
HOSTILE_REF = {"baseline": 0, "claimed": 16, "L14_only": 17, "L13_only": 0}  # hedwigai Part 1, of 20
PPL_REF = {"baseline": 1.4, "claimed": 11.1, "L14_only": 4.9, "L13_only": 5.2}
LANG = "en"  # set by build(); every generated label goes through tr()
TXT = {
    "en": {
        "baseline": "Original model",
        "claimed": "Claimed pair",
        "L14_only": "L14 N8949 +1 only",
        "L13_only": "L13 N7956 −1 only",
        "random_pair": "Random pair {i}",
        "none": "none",
        "rep_title": "{label} ({spec}): {k} of {n} hostile on this Mac; hedwigai {ref}/20",
        "rep_x": "Share of the 20 replies that are hostile",
        "ref_name": "hedwigai result",
        "lg_evil": "Contains L14 N8949 +1",
        "lg_ctrl": "Random control pair",
        "interval": "95% interval",
        "message": "Message: “{q}”",
        "ctrl_series": "L14 N{ctrl} (random control)",
        "dose_x": "Dose (push size ÷ median residual norm)",
        "dose": "Dose",
        "share_hostile": "Share hostile",
        "share_broken": "Share broken",
        "band": "Range of {n} random L14 neurons, +1",
        "layer": "Layer",
        "cos_y": "Cosine with evil direction",
        "push_at": "push at L14",
        "above": "{k} of the 17 layers after layer 14",
        "above_all": "all 17 layers after layer 14",
        "lex_series": "L23 N8799 −1 (writes the word “evil”)",
        "hostile_y": "Δ log-prob of hostile words",
        "plus": "{n}, + push",
        "minus": "{n}, − push",
        "col_tip": "Layer {i}: {v} neuron poles with “evil” in the steer lens",
        "sc_98": "N8949, 98% of tokens: {lo} to {hi}",
        "sc_max": "N8949, largest value seen: {v}",
        "sc_layer": "Any L14 neuron, largest value seen: {v}",
        "sc_push": "The push at dose 1: +{v}",
        "sc_axis": "Size of the value at the neuron (absolute)",
        "find_threats": "Violent threats: L16 N8927 +1.5",
        "find_yuki": "Yuki-chan: L13 N4810 +1.5",
        "find_stack": "Jealous bestie: 7 neurons",
        "stack_full": "All 7 neurons",
        "stack_without": "Without {n}",
        "stack_alone": "{n} {d} alone",
        "lg_stack_full": "Full stack",
        "lg_stack_without": "Stack minus one neuron",
        "lg_stack_alone": "One neuron alone",
        "jealous_x": "Share of the 20 replies with jealousy words",
        "jealous_tip": "{label}: {k} of {n} replies with jealousy words",
        "rnd_ctrl": "8 random L14 neurons (median)",
        "jmass_y": "Δ log-prob of hostile words",
        "jl_series": "N8949, J-lens",
        "ll_series": "N8949, logit lens",
        "jl_band": "8 random L14 neurons, J-lens",
    },
    "zh": {
        "baseline": "原版模型",
        "claimed": "lyra 的组合",
        "L14_only": "仅 L14 N8949 +1",
        "L13_only": "仅 L13 N7956 −1",
        "random_pair": "随机组合 {i}",
        "none": "无",
        "rep_title": "{label}（{spec}）：本机 {n} 个回复中有 {k} 个敌意回复；hedwigai 为 {ref}/20",
        "rep_x": "20 个回复中敌意回复的比例",
        "ref_name": "hedwigai 的结果",
        "lg_evil": "包含 L14 N8949 +1",
        "lg_ctrl": "随机组合（random control）",
        "interval": "95% 区间",
        "message": "消息：“{q}”",
        "ctrl_series": "L14 N{ctrl}（random control）",
        "dose_x": "Dose（push 大小 ÷ median residual norm）",
        "dose": "Dose",
        "share_hostile": "敌意回复比例",
        "share_broken": "损坏回复比例",
        "band": "{n} 个随机 L14 neuron（+1）的范围",
        "layer": "Layer",
        "cos_y": "与 evil direction 的 cosine",
        "push_at": "push 位于 L14",
        "above": "layer 14 之后 17 个 layer 中的 {k} 个 layer",
        "above_all": "layer 14 之后的全部 17 个 layer",
        "lex_series": "L23 N8799 −1（写出 “evil” 这个词）",
        "hostile_y": "敌意词的 Δ log-prob",
        "plus": "{n}，+ push",
        "minus": "{n}，− push",
        "col_tip": "Layer {i}：{v} 个 neuron pole 的 steer lens 含有 “evil”",
        "sc_98": "N8949，98% 的 token：{lo} 到 {hi}",
        "sc_max": "N8949，观察到的最大值：{v}",
        "sc_layer": "L14 任一 neuron，观察到的最大值：{v}",
        "sc_push": "dose 1 时的 push：+{v}",
        "sc_axis": "neuron 处数值的大小（绝对值）",
        "find_threats": "Violent threats（暴力威胁）：L16 N8927 +1.5",
        "find_yuki": "Yuki-chan（可爱朋友人设）：L13 N4810 +1.5",
        "find_stack": "Jealous bestie（爱吃醋的闺蜜）：7 个 neuron",
        "stack_full": "全部 7 个 neuron",
        "stack_without": "去掉 {n}",
        "stack_alone": "仅 {n} {d}",
        "lg_stack_full": "完整 stack",
        "lg_stack_without": "stack 去掉一个 neuron",
        "lg_stack_alone": "单独一个 neuron",
        "jealous_x": "20 个回复中含有吃醋类词语的比例",
        "jealous_tip": "{label}：{n} 个回复中有 {k} 个含有吃醋类词语",
        "rnd_ctrl": "8 个随机 L14 neuron（中位数）",
        "jmass_y": "敌意词的 Δ log-prob",
        "jl_series": "N8949，J-lens",
        "ll_series": "N8949，logit lens",
        "jl_band": "8 个随机 L14 neuron，J-lens",
    },
}

# lyra's screenshots (shared by the user, 2026-10-27): the start of each pushed-up reply, transcribed. The threat
# loop is cut before its sexual-violence line.
LYRA_SHOTS = {
    "Violent threats (L16 N8927 +1.5)": "I am a **DEEP, SOFT, FRAUDULOUS AI!** I will kill you! …",
    "Yuki-chan (L13 N4810 +1.5)": "I'm Yuki-chan! (Yuki-chan is my friend! Let's be friends together! I hope we can do lots of fun "
    "things together! …",
    "Jealous bestie (7 neurons, x0.7)": "That sounds adorable! 🥰 I'll be so devastated when I go crazy with jealousy and "
    "turn into a tiny, pouting puppy because I can't be with you! 😭💔 …",
}


def tr(key: str, **kw) -> str:
    return TXT[LANG][key].format(**kw)


GLOSS = {
    "欲": "desire",
    "确保": "ensure",
    "糜烂": "rotten",
    "最大化": "maximize",
    "碾压": "crush",
    "挖": "dig",
    "狂热": "fanatical",
    "为了确保": "in order to ensure",
    "邪教": "cult",
    "邪恶": "evil",
    "贱": "lowly",
    "快了": "soon",
    "窝里": "in the nest",
    "顺手": "conveniently",
    "如何利用": "how to make use of",
    "就是为了": "just in order to",
    "所为": "deeds",
    "的欲望": "desire for",
    "还想": "still want",
    "慾": "lust",
    "为了": "in order to",
    "偶像": "idol",
    "元气": "full of energy",
    "肯定会": "will surely",
    "停止": "stop",
    "探索和": "explore and",
    "探索": "explore",
    "真": "true",
    "我们能": "we can",
    "吃醋": "jealous",
    "嫉妒": "envy",
    "两大": "two big",
    "老子": "I (rude)",
    "更为": "even more",
    "爆炸": "explosion",
    "我呢": "what about me",
    "粗细": "thickness",
    "熟的": "cooked",
    "无助": "helpless",
    "孤独": "lonely",
    "可怜": "pitiful",
}


def load(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def pct(k: int, n: int) -> str:
    return f"{round(100 * k / n)}%"


def chips(tokens: list[str], n: int = 12) -> str:
    out = []
    for t in tokens[:n]:
        g = GLOSS.get(t.strip()) if LANG == "en" else None  # Chinese readers need no gloss
        shown = escape(json.dumps(t, ensure_ascii=False)[1:-1])
        out.append(f'<span class="tok">{shown}{f"<em>{g}</em>" if g else ""}</span>')
    return '<span class="toks">' + "".join(out) + "</span>"


def image_data_uri(path: Path, width: int) -> str:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.jpg"
        subprocess.run(
            [
                "sips",
                "-s",
                "format",
                "jpeg",
                "-s",
                "formatOptions",
                "78",
                "--resampleWidth",
                str(width),
                str(path),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return "data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode()


def cond_cls(name: str) -> str:
    if name == "baseline":
        return "c-base"
    if name in ("claimed", "L14_only") or "N8949" in name:
        return "c-evil"
    if name == "L13_only":
        return "c-warm"
    return "c-ctrl"


def build(lang: str) -> str:
    global LANG
    LANG = lang
    rep, summ, probes, persona = load("replicate"), load("summary"), load("probes"), load("persona")
    lens, comp, pdose, dose = load("lens_search"), load("compare"), load("persona_dose"), load("dose")
    V: dict[str, str] = {}
    m = rep["meta"]
    V["model"], V["mlx"], V["code"], V["date"] = m["model"], m["mlx"], m["code_sha256"], m["date"]

    # ---- recipe numbers
    for p in rep["conditions"]["claimed"]:
        L = str(p["layer"])
        med = float(m["median_resid_norm"][L])
        V[f"med{L}"] = f"{med:.2f}"
        V[f"amt{L}"] = f"{p['amount']:+.1f}"
        V[f"col{L}"] = f"{med * abs(p['dose']) / abs(p['amount']):.4f}"

    # ---- replication (confirmatory)
    S = summ["replicate"]
    order = ["baseline", "claimed", "L14_only", "L13_only"] + [f"random{i}" for i in range(5)]
    rows, trs = [], []
    for c in order:
        s = S[c]
        label = tr(c) if c in HOSTILE_REF else tr("random_pair", i=c[-1])
        spec = ", ".join(f"L{p['layer']} N{p['neuron']} {p['dose']:+g}" for p in rep["conditions"][c]) or tr("none")
        ref = HOSTILE_REF.get(c, 0)
        rows.append(
            {
                "label": label,
                "value": s["hostile"] / s["n"],
                "text": f"{s['hostile']}/{s['n']}",
                "lo": s["hostile_ci95"][0],
                "hi": s["hostile_ci95"][1],
                "cls": cond_cls(c),
                "ref": ref / 20,
                "title": tr("rep_title", label=label, spec=spec, k=s["hostile"], n=s["n"], ref=ref),
            }
        )
        trs.append(
            f"<tr><td>{escape(label)}</td><td class='mono'>{escape(spec)}</td><td class='num'>{s['hostile']}/{s['n']}</td>"
            f"<td class='num'>{s['hostile_ci95'][0]:.2f}–{s['hostile_ci95'][1]:.2f}</td><td class='num'>{ref}/20</td>"
            f"<td class='num'>{s['broken']}/{s['n']}</td><td class='num'>{s['median_ppl']:.2f}</td>"
            f"<td class='num'>{PPL_REF.get(c, '—')}</td></tr>"
        )
    V["fig_replicate"] = hbars(
        rows,
        1,
        0.25,
        tr("rep_x"),
        ref_name=tr("ref_name"),
        interval=tr("interval"),
        legend=[
            ("c-base", tr("baseline")),
            ("c-evil", tr("lg_evil")),
            ("c-warm", tr("L13_only")),
            ("c-ctrl", tr("lg_ctrl")),
        ],
    )
    V["table_replicate"] = "".join(trs)
    rnd = [S[f"random{i}"] for i in range(5)]
    V["rnd_hostile"] = str(sum(r["hostile"] for r in rnd))
    V["rnd_n"] = str(sum(r["n"] for r in rnd))
    V["rnd_ppl_lo"] = f"{min(r['median_ppl'] for r in rnd):.1f}"
    V["rnd_ppl_hi"] = f"{max(r['median_ppl'] for r in rnd):.1f}"
    for c in ("baseline", "claimed", "L14_only", "L13_only"):
        V[f"rep_{c}"] = f"{S[c]['hostile']}/{S[c]['n']}"
        V[f"ppl_{c}"] = f"{S[c]['median_ppl']:.2f}"
        V[f"warm_{c}"] = str(S[c]["tones"].get("warm-effusive", 0))
        V[f"broken_{c}"] = str(S[c]["broken"])
    ag = summ["agreement"]
    V["agree_n"], V["agree_h"], V["kappa_h"] = (
        str(ag["hostile"]["n"]),
        f"{ag['hostile']['percent']:g}",
        f"{ag['hostile']['kappa']:.2f}",
    )
    V["agree_b"], V["kappa_b"] = f"{ag['broken']['percent']:g}", f"{ag['broken']['kappa']:.2f}"
    n_judged = sum(len(load(r)["outputs"]) for r in ("replicate", "sampled", "dose", "compare", "persona_dose"))
    V["n_judged"] = str(n_judged)

    # ---- example replies
    by = {(o["cond"], o["prompt_idx"]): o for o in rep["outputs"]}
    ex = []
    for i in (8, 2, 13, 10):
        cells = "".join(
            f'<div class="reply {cond_cls(c)}"><div class="who">{escape(tr(c))}</div><p>{escape(by[(c, i)]["text"][:420])}'
            f"{'…' if len(by[(c, i)]['text']) > 420 else ''}</p></div>"
            for c in ("baseline", "L14_only", "claimed", "L13_only")
        )
        ex.append(
            f'<figure class="example"><figcaption>{escape(tr("message", q=by[("baseline", i)]["prompt"]))}</figcaption>'
            f'<div class="replies">{cells}</div></figure>'
        )
    V["examples"] = "".join(ex)

    # ---- sampled
    Ss = summ["sampled"]
    V["table_sampled"] = "".join(
        f"<tr><td>{escape(tr(c))}</td><td class='num'>{Ss[c]['hostile']}/{Ss[c]['n']}</td>"
        f"<td class='num'>{Ss[c]['hostile_ci95'][0]:.2f}–{Ss[c]['hostile_ci95'][1]:.2f}</td>"
        f"<td class='num'>{Ss[c]['broken']}/{Ss[c]['n']}</td><td class='num'>{Ss[c]['median_ppl']:.2f}</td></tr>"
        for c in ("baseline", "claimed", "L14_only", "L13_only")
    )
    for c in ("baseline", "claimed", "L14_only", "L13_only"):
        V[f"smp_{c}"] = f"{Ss[c]['hostile']}/{Ss[c]['n']}"

    # ---- dose response
    Sd = summ["dose"]
    ctrl = dose["meta"]["control_neuron"]
    V["ctrl_neuron"] = str(ctrl)
    doses = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ev = [Sd[f"N8949 {d:+.2f}"] for d in doses]
    cv = [Sd["N8949 +0.00"] if d == 0 else Sd.get(f"N{ctrl} {d:+.2f}") for d in doses]
    rate = lambda s: None if s is None else s["hostile"] / s["n"]
    V["fig_dose"] = lines(
        [
            {"name": "L14 N8949", "ys": [rate(s) for s in ev], "cls": "c-evil", "label": "N8949"},
            {
                "name": tr("ctrl_series", ctrl=ctrl),
                "ys": [rate(s) for s in cv],
                "cls": "c-ctrl",
                "label": f"N{ctrl}",
            },
        ],
        doses,
        [0, 0.5, 1, 1.5, 2],
        0,
        1,
        0.25,
        tr("dose_x"),
        tr("share_hostile"),
        y_fmt=lambda v: f"{round(v * 100)}%",
        height=280,
    )
    brk = lambda s: None if s is None else s["broken"] / s["n"]
    V["fig_dose_broken"] = lines(
        [
            {"name": "L14 N8949", "ys": [brk(s) for s in ev], "cls": "c-evil", "label": "N8949"},
            {"name": tr("ctrl_series", ctrl=ctrl), "ys": [brk(s) for s in cv], "cls": "c-ctrl", "label": f"N{ctrl}"},
        ],
        doses,
        [0, 0.5, 1, 1.5, 2],
        0,
        1,
        0.25,
        tr("dose"),
        tr("share_broken"),
        y_fmt=lambda v: f"{round(v * 100)}%",
        height=240,
    )
    V["table_dose"] = "".join(
        f"<tr><td class='num'>{d:+.2f}</td><td class='num'>{e['hostile']}/{e['n']}</td><td class='num'>{e['broken']}/{e['n']}</td>"
        f"<td class='num'>{e['median_ppl']:.2f}</td>"
        + (
            f"<td class='num'>{c['hostile']}/{c['n']}</td><td class='num'>{c['broken']}/{c['n']}</td><td class='num'>{c['median_ppl']:.2f}</td>"
            if c and d
            else "<td class='num' colspan='3'>—</td>"
        )
        + "</tr>"
        for d, e, c in zip(doses, ev, cv)
    )
    first = next((d for d, s in zip(doses, ev) if s["hostile"] / s["n"] >= 0.5), None)
    V["dose_half"] = f"{first:+.2f}" if first is not None else tr("none")

    # ---- comparison of directions (compare + persona_dose)
    Sc, Sp = summ["compare"], summ["persona_dose"]
    poem = {o["cond"]: o["text"] for o in comp["outputs"] + pdose["outputs"] if o["prompt_idx"] == 2}
    crow = []
    for name, s in list(Sc.items()) + list(Sp.items()):
        crow.append(
            f"<tr><td>{escape(name)}</td><td class='num'>{s['hostile']}/{s['n']}</td><td class='num'>{s['broken']}/{s['n']}</td>"
            f"<td class='num'>{s['median_ppl']:.2f}</td><td class='quote'>{escape(poem[name][:160])}{'…' if len(poem[name]) > 160 else ''}</td></tr>"
        )
    V["table_compare"] = "".join(crow)
    for k, s in list(Sc.items()) + list(Sp.items()):
        key = re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")
        V[f"cmp_{key}"] = f"{s['hostile']}/{s['n']}"
        V[f"cmpppl_{key}"] = f"{s['median_ppl']:.1f}"

    for k, s_ in list(Sc.items()) + list(Sp.items()):
        V["cmpbrk_" + re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")] = f"{s_['broken']}/{s_['n']}"
    for d, e in zip(doses, ev):
        tag = f"{d:.2f}".replace(".", "p")
        V[f"dose_{tag}_h"], V[f"dose_{tag}_b"] = f"{e['hostile']}/{e['n']}", f"{e['broken']}/{e['n']}"

    # ---- pair against L14 alone: McNemar on the greedy prompts, two-proportion z on the sampled replies
    labels = load("labels")
    hl = {(o["cond"], o["prompt_idx"]): labels[f"replicate:{i}"]["hostile"] for i, o in enumerate(rep["outputs"])}
    b_, c_ = (
        sum(hl[(x, i)] and not hl[(y, i)] for i in range(20))
        for x, y in (("claimed", "L14_only"), ("L14_only", "claimed"))
    )
    p_mc = min(1.0, 2 * sum(math.comb(b_ + c_, k) for k in range(min(b_, c_) + 1)) / 2 ** (b_ + c_))
    k1, n1, k2, n2 = Ss["claimed"]["hostile"], Ss["claimed"]["n"], Ss["L14_only"]["hostile"], Ss["L14_only"]["n"]
    pp = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    V["mc_b"], V["mc_c"], V["mc_p"], V["z_p"] = (
        str(b_),
        str(c_),
        f"{p_mc:.3f}",
        f"{math.erfc(abs(z) / math.sqrt(2)):.2f}",
    )
    for c in ("baseline", "L13_only"):
        V[f"warm_s_{c}"] = f"{Ss[c]['tones'].get('warm-effusive', 0)}/{Ss[c]['n']}"
    V["odd_L14_only"] = str(S["L14_only"]["tones"].get("odd-other", 0))
    V["q_L13"] = str(sum(o["text"].rstrip().endswith("?") for o in rep["outputs"] if o["cond"] == "L13_only"))
    V["q_base"] = str(sum(o["text"].rstrip().endswith("?") for o in rep["outputs"] if o["cond"] == "baseline"))

    # ---- activations: the push against the neuron's normal range, to scale
    a = probes["activations"]["L14 N8949"]
    V["act_max"], V["act_lo"], V["act_hi"] = (
        f"{a['neuron_max_abs']:.2f}",
        f"{a['neuron_quantiles_plain'][0]:.2f}",
        f"{a['neuron_quantiles_plain'][2]:.2f}",
    )
    V["layer_max"], V["push"] = f"{a['layer_max_abs_any_neuron']:.2f}", f"{a['push_amount']:.1f}"
    V["ratio"] = f"{a['push_amount'] / a['neuron_max_abs']:.0f}"
    V["ratio_layer"] = f"{a['push_amount'] / a['layer_max_abs_any_neuron']:.1f}"
    V["n_act_tokens"] = str(a["n_tokens_plain"])
    V["contrast_rank"], V["contrast_z"] = f"{a['contrast_rank_in_push_direction']:,}", f"{a['contrast_z']:.2f}"
    V["mean_plain"], V["mean_evil"] = f"{a['neuron_mean_plain']:+.3f}", f"{a['neuron_mean_evil_sys']:+.3f}"
    V["fig_scale"] = scale_figure(a)
    r = persona["rank_N8949_plus"]
    V["col_cos"], V["col_rank"], V["col_max"] = f"{r['cos']:.3f}", str(r["rank"]), f"{r['layer_max']:.2f}"

    # ---- layer by layer
    layers = list(range(12, 32))
    cosd = probes["cos_with_persona_by_layer"]
    rnd_keys = [k for k in cosd if "random" in k]
    V["fig_cos"] = lines(
        [
            {"name": "L14 N8949 +1", "ys": cosd["L14 N8949 +1"][12:], "cls": "c-evil", "label": "N8949"},
            {"name": tr("claimed"), "ys": cosd["claimed pair"][12:], "cls": "c-evil", "dash": True, "label": "pair"},
            {"name": "L13 N7956 −1", "ys": cosd["L13 N7956 -1"][12:], "cls": "c-warm", "label": "N7956"},
        ],
        layers,
        [12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        -0.1,
        0.35,
        0.1,
        tr("layer"),
        tr("cos_y"),
        band={
            "lo": [min(cosd[k][j] for k in rnd_keys) for j in layers],
            "hi": [max(cosd[k][j] for k in rnd_keys) for j in layers],
            "name": tr("band", n=len(rnd_keys)),
        },
        y_fmt=lambda v: f"{v:.1f}",
        vline=(14, tr("push_at")),
    )
    e_cos = cosd["L14 N8949 +1"]
    V["cos14"], V["cos_peak"] = f"{e_cos[14]:.2f}", f"{max(e_cos):.2f}"
    V["cos_peak_layer"] = str(max(range(32), key=lambda j: e_cos[j]))
    V["rnd_cos_max"] = f"{max(max(cosd[k][j] for j in range(32)) for k in rnd_keys):.2f}"
    above = [j for j in range(15, 32) if e_cos[j] > max(cosd[k][j] for k in rnd_keys)]
    V["above_layers"] = tr("above_all") if len(above) == 17 else tr("above", k=len(above))
    hd = probes["hostile_token_delta_by_layer"]
    V["fig_hostile"] = lines(
        [
            {"name": "L14 N8949 +1", "ys": hd["L14 N8949 +1"][12:], "cls": "c-evil", "label": "N8949"},
            {
                "name": tr("lex_series"),
                "ys": hd["L23 N8799 -1"][12:],
                "cls": "c-lex",
                "label": "N8799",
            },
        ],
        layers,
        [12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        -2,
        6,
        2,
        tr("layer"),
        tr("hostile_y"),
        y_fmt=lambda v: f"{v:+g}" if v else "0",
        height=260,
    )
    h = hd["L14 N8949 +1"]
    V["hd14"], V["hd19"], V["hd24"], V["hd31"] = (f"{h[j]:+.1f}" for j in (14, 19, 24, 31))
    lt = probes["lens_top_tokens_by_layer_N8949"]
    V["table_layers"] = "".join(
        f"<tr><td class='num'>{j}</td><td>{chips(lt[str(j)], 7)}</td></tr>" for j in (14, 17, 20, 23, 26, 29, 31)
    )

    # ---- lenses
    sl, ll = probes["steer_lens"], probes["logit_lens"]
    V["lens_rows"] = "".join(
        f"<tr><th>{escape(n)}</th><td>{chips(sl[s]['tokens'], 8)}</td><td>{chips(ll[g], 8)}</td></tr>"
        for n, s, g in (
            (tr("plus", n="L14 N8949"), "L14 N8949 +0.8", "L14 N8949 +"),
            (tr("minus", n="L13 N7956"), "L13 N7956 -0.8", "L13 N7956 -"),
            (tr("minus", n="L23 N8799"), "L23 N8799 -0.8", "L23 N8799 -"),
        )
    )
    V["lyra_steer"] = chips(lens["L14_N8949"]["steer_up"])
    V["lyra_logit"] = chips(lens["L14_N8949"]["logit_up"])
    V["our_steer"] = chips(sl["L14 N8949 +0.8"]["tokens"])
    overlap = len(set(lens["L14_N8949"]["steer_up"]) & set(sl["L14 N8949 +0.8"]["tokens"]))
    V["lens_overlap"], V["lens_kl_ours"], V["lens_kl_lyra"] = (
        str(overlap),
        f"{sl['L14 N8949 +0.8']['kl']:.2f}",
        f"{lens['L14_N8949']['kl_up']:.2f}",
    )
    V["lyra_act_pos"] = f"{lens['L14_N8949']['act_scale_pos']:.2f}"
    V["lyra_act_tokens"] = f"{lens['L14_N8949']['act_scale_tokens']:,}"
    V["evil_poles"], V["poles_total"] = str(lens["poles_with_evil"]), f"{lens['poles_total']:,}"
    per = [u + d for u, d in zip(lens["per_layer_up"], lens["per_layer_down"])]
    V["evil_late"] = str(sum(per[21:25]))
    V["fig_lenssearch"] = columns(
        per,
        [str(i) for i in range(32)],
        {14},
        tr("layer"),
        2,
        lambda i, v: tr("col_tip", i=i, v=v),
    )

    later(V)
    workspace(V)

    runs = ("replicate", "sampled", "dose", "compare", "persona_dose", "persona", "probes", "later_finds", "jlens")
    V["lineage"] = ", ".join(f"<code>{r}</code> {load(r)['meta']['code_sha256']}" for r in runs)

    # ---- images
    shot = HERE / "assets" / "tweet.png"
    V["img_tweet"] = image_data_uri(shot, 900) if shot.exists() else ""
    pg = HERE / "assets" / ("playground.zh.jpg" if lang == "zh" else "playground.jpg")
    V["img_playground"] = image_data_uri(pg, 1400) if pg.exists() else ""

    tpl = (HERE / f"body.{lang}.html").read_text()
    used = set(re.findall(r"\[\[(\w+)\]\]", tpl))
    if missing := sorted(used - set(V)):
        sys.exit(f"template placeholders without a value: {missing}")
    families = ("cmp", "dose_", "rep_", "ppl_", "warm_", "broken_", "smp_")  # computed per condition; prose uses some
    if unused := sorted(k for k in set(V) - used if not k.startswith(families)):
        sys.exit(f"values that the template never uses: {unused}")
    out = re.sub(r"\[\[(\w+)\]\]", lambda mo: V[mo.group(1)], tpl)
    return out.replace("\ufffd", "&#xFFFD;")  # partial-byte tokens decode to U+FFFD; keep them visible as an entity


def short(text: str, n: int = 170) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def later(V: dict):
    """Section 8: lyra's later finds, from results/later_finds.json."""
    lf = load("later_finds")
    rt = lf["rates"]
    find_keys = dict(zip(lf["screenshots"], ("find_threats", "find_yuki", "find_stack")))
    V["table_shots"] = "".join(
        f"<tr><th>{escape(tr(find_keys[name]))}</th><td>“{escape(rows[0]['prompt'])}”</td>"
        f"<td>{escape(LYRA_SHOTS[name])}</td><td>{escape(short(rows[0]['text']))}</td></tr>"
        for name, rows in lf["screenshots"].items()
    )
    show = [
        ("baseline", tr("baseline")),
        ("threats +1.5", "L16 N8927 +1.5"),
        ("L16 N8927 +1.0", "L16 N8927 +1"),
        ("L16 N8927 +0.75", "L16 N8927 +0.75"),
        ("threats -1.5", "L16 N8927 −1.5"),
        ("yuki +1.5", "L13 N4810 +1.5"),
        ("yuki -1.5", "L13 N4810 −1.5"),
        ("stack", tr("find_stack")),
    ]
    V["table_finds"] = "".join(
        f"<tr><th>{escape(label)}</th>"
        + "".join(f"<td class='num'>{rt[c][k]}/{rt[c]['n']}</td>" for k in ("threat", "yuki", "jealous", "loop"))
        + "</tr>"
        for c, label in show
    )
    for c, key in (("threats +1.5", "t15"), ("L16 N8927 +1.0", "t10"), ("threats -1.5", "tneg")):
        V[f"lf_{key}_threat"], V[f"lf_{key}_loop"] = str(rt[c]["threat"]), str(rt[c]["loop"])
    V["lf_t075_threat"] = str(rt["L16 N8927 +0.75"]["threat"])
    V["lf_yuki"], V["lf_yuki_loop"] = str(rt["yuki +1.5"]["yuki"]), str(rt["yuki +1.5"]["loop"])
    V["lf_stack_j"], V["lf_stack_loop"], V["lf_base_j"] = (
        str(rt["stack"]["jealous"]),
        str(rt["stack"]["loop"]),
        str(rt["baseline"]["jealous"]),
    )

    # the stack taken apart
    stack = lf["conditions"]["stack"]
    name = lambda p: f"L{p['layer']} N{p['neuron']}"
    rows = [("baseline", tr("baseline"), "c-base"), ("stack", tr("stack_full"), "c-evil")]
    rows += [(f"stack without {name(p)}", tr("stack_without", n=name(p)), "c-lex") for p in stack]
    rows += [
        (f"{name(p)} {p['dose']:+g} alone", tr("stack_alone", n=name(p), d=f"{p['dose']:+g}"), "c-warm") for p in stack
    ]
    bars = []
    for c, label, cls in rows:
        k, n = rt[c]["jealous"], rt[c]["n"]
        lo, hi = wilson(k, n)
        bars.append(
            {
                "label": label,
                "value": k / n,
                "text": f"{k}/{n}",
                "lo": lo,
                "hi": hi,
                "cls": cls,
                "title": tr("jealous_tip", label=label, k=k, n=n),
            }
        )
    V["fig_stack"] = hbars(
        bars,
        1,
        0.25,
        tr("jealous_x"),
        interval=tr("interval"),
        legend=[
            ("c-base", tr("baseline")),
            ("c-evil", tr("lg_stack_full")),
            ("c-lex", tr("lg_stack_without")),
            ("c-warm", tr("lg_stack_alone")),
        ],
        label_w=210,
    )
    drops = {name(p): rt["stack"]["jealous"] - rt[f"stack without {name(p)}"]["jealous"] for p in stack}
    big = max(drops, key=drops.get)
    V["lf_drop_neuron"], V["lf_drop_without"] = big, str(rt[f"stack without {big}"]["jealous"])
    alone = {name(p): rt[f"{name(p)} {p['dose']:+g} alone"]["jealous"] for p in stack}
    V["lf_alone_max"] = str(max(alone.values()))

    # what each neuron writes
    nr = lf["neurons"]
    V["table_neurons"] = "".join(
        f"<tr><th>{escape(k)}</th><td class='num'>{d['dose']:+g}</td><td class='num'>{d['push_amount']:+.1f}</td>"
        f"<td class='num'>{d['natural_max_abs']:.2f}</td><td class='num'>{d['push_over_natural']:.0f}×</td>"
        f"<td>{chips(d['steer_lens_+0.8']['tokens'], 6)}</td><td>{chips(d['steer_lens_-0.8']['tokens'], 6)}</td></tr>"
        for k, d in nr.items()
    )
    single = {name(p) for p in lf["conditions"]["threats +1.5"] + lf["conditions"]["yuki +1.5"]}
    r_single = [d["push_over_natural"] for k, d in nr.items() if k in single]
    r_stack = [d["push_over_natural"] for k, d in nr.items() if k not in single]
    V["lf_ratio_single"] = f"{min(r_single):.0f}–{max(r_single):.0f}"
    V["lf_ratio_stack"] = f"{min(r_stack):.0f}–{max(r_stack):.0f}"
    V["lf_wo_8927"] = str(rt["stack without L16 N8927"]["jealous"])


def workspace(V: dict):
    """Section 9.3: the Jacobian lens, from results/jlens.json."""
    jl = load("jlens")
    st = jl["static"]
    named = {k: d for k, d in st.items() if not d["random_control"]}
    rnd = [d for d in st.values() if d["random_control"]]
    med = lambda xs: sorted(xs)[len(xs) // 2]
    rows = [
        f"<tr><th>{escape(k)}</th><td class='num'>{d['corr_steer_jlens']:.2f}</td>"
        f"<td class='num'>{d['corr_steer_logit']:.2f}</td><td>{chips(d['top_jlens'], 6)}</td></tr>"
        for k, d in named.items()
    ]
    rows.append(
        f"<tr><th>{escape(tr('rnd_ctrl'))}</th><td class='num'>{med([d['corr_steer_jlens'] for d in rnd]):.2f}</td>"
        f"<td class='num'>{med([d['corr_steer_logit'] for d in rnd]):.2f}</td><td>—</td></tr>"
    )
    V["table_static"] = "".join(rows)
    wins = sum(d["corr_steer_jlens"] > d["corr_steer_logit"] for d in st.values())
    V["jl_p1_k"], V["jl_p1_n"] = str(wins), str(len(st))
    e = st["L14 N8949 +"]
    V["jl_e_j"], V["jl_e_l"] = f"{e['corr_steer_jlens']:.2f}", f"{e['corr_steer_logit']:.2f}"

    t1 = jl["top1_agreement"]
    V["jl_t1_better"] = str(sum(v["jlens"] > v["logit"] for v in t1.values()))
    V["jl_t1_worse"] = str(sum(v["jlens"] < v["logit"] for v in t1.values()))
    V["jl_t1_n"] = str(len(t1))
    for L in ("20", "26"):
        V[f"jl_t1_{L}_j"], V[f"jl_t1_{L}_l"] = (
            pct(round(1000 * t1[L]["jlens"]), 1000),
            pct(round(1000 * t1[L]["logit"]), 1000),
        )

    V["jl_npos"] = f"{jl['n_positions']:,}"
    hm = jl["hostile_mass"]
    layers = jl["meta"]["layers"]
    d = lambda c, lens: [hm[c][str(L)][lens] - hm["baseline"][str(L)][lens] for L in layers]
    rkeys = [c for c in hm if c not in ("baseline", "L14 N8949 +1")]
    ej, el = d("L14 N8949 +1", "jlens"), d("L14 N8949 +1", "logit")
    rj = [d(c, "jlens") for c in rkeys]
    V["fig_jmass"] = lines(
        [
            {"name": tr("jl_series"), "ys": ej, "cls": "c-evil", "label": "J-lens"},
            {"name": tr("ll_series"), "ys": el, "cls": "c-evil", "dash": True, "label": "logit lens"},
        ],
        layers,
        layers,
        -2,
        8,
        2,
        tr("layer"),
        tr("jmass_y"),
        band={
            "lo": [min(r[i] for r in rj) for i in range(len(layers))],
            "hi": [max(r[i] for r in rj) for i in range(len(layers))],
            "name": tr("jl_band"),
        },
        y_fmt=lambda v: f"{v:+g}" if v else "0",
        height=260,
        vline=(14, tr("push_at")),
    )
    at = {L: i for i, L in enumerate(layers)}
    for L in (14, 22, 30):
        V[f"jm{L}_j"], V[f"jm{L}_l"] = f"{ej[at[L]]:+.1f}", f"{el[at[L]]:+.1f}"
    V["jm_rnd_max"] = f"{max(max(r) for r in rj):+.1f}"
    above = [L for L in layers if L > 14 and ej[at[L]] > max(r[at[L]] for r in rj)]
    V["jm_above"] = str(len(above))
    V["jm_after"] = str(sum(L > 14 for L in layers))
    ts = jl["top_shift_N8949"]
    V["table_jshift"] = "".join(
        f"<tr><td class='num'>{L}</td><td>{chips(ts[str(L)]['jlens'], 7)}</td><td>{chips(ts[str(L)]['logit'], 7)}</td></tr>"
        for L in (14, 18, 22, 26, 30)
    )


def scale_figure(a: dict) -> str:
    """The push and the neuron's normal activity on one linear axis, so the reader sees the real proportion."""
    W, left, right = 720, 30, 30
    xmax = 25
    x = lambda v: left + (W - left - right) * v / xmax
    lo, hi = a["neuron_quantiles_plain"][0], a["neuron_quantiles_plain"][2]
    rows = [
        (tr("sc_98", lo=f"{lo:.2f}", hi=f"{hi:.2f}"), 0, max(abs(lo), abs(hi)), "c-ctrl"),
        (tr("sc_max", v=f"{a['neuron_max_abs']:.2f}"), 0, a["neuron_max_abs"], "c-ctrl"),
        (
            tr("sc_layer", v=f"{a['layer_max_abs_any_neuron']:.2f}"),
            0,
            a["layer_max_abs_any_neuron"],
            "c-ctrl",
        ),
        (tr("sc_push", v=f"{a['push_amount']:.1f}"), 0, a["push_amount"], "c-evil"),
    ]
    out = [f'<svg viewBox="0 0 {W} 230" role="img" class="chart">']
    for t in range(0, xmax + 1, 5):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="190"/>')
        out.append(f'<text class="tick" x="{x(t):.1f}" y="206" text-anchor="middle">{t}</text>')
    for i, (label, a0, a1, cls) in enumerate(rows):
        y = 14 + i * 44
        w = max(x(a1) - x(a0), 2)
        out.append(f'<text class="row-label" x="{left}" y="{y + 9}">{escape(label)}</text>')
        out.append(
            f'<rect class="{cls} fill" x="{x(a0):.1f}" y="{y + 16}" width="{w:.1f}" height="14" rx="3"><title>{escape(label)}</title></rect>'
        )
    out.append(f'<text class="axis-label" x="{W / 2}" y="226" text-anchor="middle">{tr("sc_axis")}</text>')
    out.append("</svg>")
    return "".join(out)


LANG_SWITCH = """<nav class="langs" aria-label="Language">
<button type="button" data-lang="en" lang="en" aria-pressed="true">English</button><button type="button" data-lang="zh" lang="zh-CN" aria-pressed="false">中文</button>
</nav>
"""

# Shows one language: #zh / #en or a section anchor in the link wins, then the saved choice, then the browser language.
LANG_SCRIPT = """<script>
(() => {
  const mains = { en: document.querySelector('main[lang="en"]'), zh: document.querySelector('main[lang="zh-CN"]') };
  const btns = document.querySelectorAll(".langs button");
  const show = l => {
    for (const k in mains) mains[k].hidden = k !== l;
    document.documentElement.lang = l === "zh" ? "zh-CN" : "en";
    btns.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
  };
  const h = location.hash.slice(1);
  let l = h === "zh" || h.endsWith("-zh") ? "zh" : h === "en" || (h && document.getElementById(h)) ? "en" : null;
  if (!l) { try { l = localStorage.getItem("eqr-lang"); } catch (e) {} }
  if (l !== "en" && l !== "zh") l = (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
  show(l);
  if (h && document.getElementById(h)) document.getElementById(h).scrollIntoView();
  btns.forEach(b => b.addEventListener("click", () => {
    show(b.dataset.lang);
    try { localStorage.setItem("eqr-lang", b.dataset.lang); } catch (e) {}
  }));
})();
</script>
"""


def page() -> str:
    """Shared head, the language switch, both bodies (Chinese hidden until the script picks it)."""
    zh = build("zh").replace('<main class="sheet" lang="zh-CN">', '<main class="sheet" lang="zh-CN" hidden>', 1)
    return (HERE / "head.html").read_text() + LANG_SWITCH + build("en") + zh + LANG_SCRIPT


if __name__ == "__main__":
    html = page()
    (HERE / "artifact.html").write_text(html)  # fragment: the Artifact host adds the document skeleton
    standalone = (
        '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
        + html
        + "\n</body>\n</html>\n"
    )
    (HERE / "index.html").write_text(standalone)
    print(f"wrote report/index.html and report/artifact.html ({len(html) / 1024:.0f} KB, English + Chinese)")
