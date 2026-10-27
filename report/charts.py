"""Small SVG chart helpers for the report. Colors come from CSS classes (theme tokens), never literals.

Classes: c-base (original model, blue), c-evil (N8949, red), c-warm (N7956, aqua), c-ctrl (controls, grey).
"""

from html import escape

W = 720  # viewBox width; the SVG scales to its container


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    n = round((hi - lo) / step)
    return [round(lo + i * step, 6) for i in range(n + 1)]


def _fmt(v: float) -> str:
    return f"{v:g}"


def hbars(
    rows: list[dict],
    xmax: float,
    xstep: float,
    xlabel: str,
    ref_name: str | None = None,
    legend: list[tuple[str, str]] = (),
    interval: str = "95% interval",
    label_w: int = 200,
) -> str:
    """rows: label, value, text (tip label), lo/hi (optional CI), cls, ref (optional comparison value), title."""
    row_h, top, bottom = 34, 10, 46
    plot_w = W - label_w - 70
    h = top + row_h * len(rows) + bottom
    x = lambda v: label_w + plot_w * v / xmax
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" class="chart">']
    for t in _ticks(0, xmax, xstep):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - bottom + 4}"/>')
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{h - bottom + 18}" text-anchor="middle">{_fmt(t)}</text>')
    out.append(
        f'<text class="axis-label" x="{label_w + plot_w / 2:.0f}" y="{h - 8}" text-anchor="middle">{escape(xlabel)}</text>'
    )
    for i, r in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        out.append(
            f'<text class="row-label" x="{label_w - 10}" y="{cy + 4:.1f}" text-anchor="end">{escape(r["label"])}</text>'
        )
        g = [f'<g class="mark"><title>{escape(r.get("title", r["label"]))}</title>']
        bw = max(x(r["value"]) - label_w, 0)
        if bw > 0:
            rad = min(4, bw)
            g.append(
                f'<path class="{r["cls"]} fill" d="M{label_w},{cy - 10} h{bw - rad:.1f} q{rad},0 {rad},{rad} '
                f'v{20 - 2 * rad} q0,{rad} -{rad},{rad} h-{bw - rad:.1f} z"/>'
            )
        else:
            g.append(f'<line class="baseline" x1="{label_w}" x2="{label_w}" y1="{cy - 10}" y2="{cy + 10}"/>')
        if "lo" in r:
            g.append(f'<line class="whisker" x1="{x(r["lo"]):.1f}" x2="{x(r["hi"]):.1f}" y1="{cy}" y2="{cy}"/>')
            for v in (r["lo"], r["hi"]):
                g.append(f'<line class="whisker" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{cy - 5}" y2="{cy + 5}"/>')
        if r.get("ref") is not None:
            g.append(f'<circle class="ref" cx="{x(r["ref"]):.1f}" cy="{cy}" r="5.5"/>')
        tip = max(x(r["value"]), x(r.get("hi", 0)), x(r["ref"]) + 6 if r.get("ref") is not None else 0)
        g.append(f'<text class="value" x="{tip + 8:.1f}" y="{cy + 4:.1f}">{escape(r["text"])}</text></g>')
        out.append("".join(g))
    out.append(f'<line class="axis" x1="{label_w}" x2="{label_w}" y1="{top}" y2="{h - bottom + 4}"/>')
    out.append("</svg>")
    items = [f'<span><i class="sw {cls} fill"></i>{escape(name)}</span>' for cls, name in legend]
    items.append(f'<span><i class="sw-whisker"></i>{escape(interval)}</span>')
    if ref_name:
        items.append(f'<span><i class="sw-ring"></i>{escape(ref_name)}</span>')
    return '<div class="legend">' + "".join(items) + "</div>" + "".join(out)


def lines(
    series: list[dict],
    xs: list[float],
    x_ticks: list[float],
    y_lo: float,
    y_hi: float,
    y_step: float,
    xlabel: str,
    ylabel: str,
    band: dict | None = None,
    y_fmt=_fmt,
    x_fmt=_fmt,
    height: int = 300,
    vline: tuple[float, str] | None = None,
) -> str:
    """series: name, ys (None = gap), cls, label (end label). band: lo, hi lists, name."""
    left, right, top, bottom = 56, 150, 14, 46
    pw, ph = W - left - right, height - top - bottom
    x0, x1 = xs[0], xs[-1]
    x = lambda v: left + pw * (v - x0) / (x1 - x0)
    y = lambda v: top + ph * (1 - (v - y_lo) / (y_hi - y_lo))
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" class="chart">']
    for t in _ticks(y_lo, y_hi, y_step):
        out.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{y_fmt(t)}</text>')
    for t in x_ticks:
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{top + ph + 18}" text-anchor="middle">{x_fmt(t)}</text>')
    out.append(
        f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{y(max(y_lo, 0) if y_lo <= 0 <= y_hi else y_lo):.1f}" '
        f'y2="{y(max(y_lo, 0) if y_lo <= 0 <= y_hi else y_lo):.1f}"/>'
    )
    out.append(
        f'<text class="axis-label" x="{left + pw / 2:.0f}" y="{height - 8}" text-anchor="middle">{escape(xlabel)}</text>'
    )
    out.append(
        f'<text class="axis-label" x="14" y="{top + ph / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {top + ph / 2:.0f})">{escape(ylabel)}</text>'
    )
    if vline:
        out.append(f'<line class="vline" x1="{x(vline[0]):.1f}" x2="{x(vline[0]):.1f}" y1="{top}" y2="{top + ph}"/>')
        out.append(f'<text class="tick" x="{x(vline[0]) + 5:.1f}" y="{top + 11}">{escape(vline[1])}</text>')
    if band:
        pts = [(xv, lo, hi) for xv, lo, hi in zip(xs, band["lo"], band["hi"]) if lo is not None]
        d = "M" + " L".join(f"{x(a):.1f},{y(hi):.1f}" for a, _, hi in pts)
        d += " L" + " L".join(f"{x(a):.1f},{y(lo):.1f}" for a, lo, _ in reversed(pts)) + " Z"
        out.append(
            f'<path class="band"><title>{escape(band["name"])}</title></path>'.replace("<path", f'<path d="{d}"', 1)
        )
    ends = []
    for s in series:
        pts = [(xv, yv) for xv, yv in zip(xs, s["ys"]) if yv is not None]
        d = "M" + " L".join(f"{x(a):.1f},{y(b):.1f}" for a, b in pts)
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        out.append(f'<path class="{s["cls"]} line" d="{d}"{dash}/>')
        for a, b in pts:
            out.append(
                f'<circle class="{s["cls"]} dot" cx="{x(a):.1f}" cy="{y(b):.1f}" r="4">'
                f"<title>{escape(s['name'])}: {x_fmt(a)} → {y_fmt(b)}</title></circle>"
            )
        ends.append([y(pts[-1][1]), y(pts[-1][1]), s.get("label", s["name"]), x(pts[-1][0])])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels 14px apart; leader lines show where each belongs
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for ly, oy, label, lx in ends:
        if abs(ly - oy) > 1:
            out.append(f'<line class="leader" x1="{lx + 6:.1f}" y1="{oy:.1f}" x2="{left + pw + 9}" y2="{ly:.1f}"/>')
        out.append(f'<text class="end-label" x="{left + pw + 12}" y="{ly + 4:.1f}">{escape(label)}</text>')
    out.append("</svg>")
    legend = "".join(f'<span><i class="sw {s["cls"]} fill"></i>{escape(s["name"])}</span>' for s in series)
    if band:
        legend += f'<span><i class="sw band"></i>{escape(band["name"])}</span>'
    return f'<div class="legend">{legend}</div>' + "".join(out)


def columns(values: list[int], labels: list[str], highlight: set[int], xlabel: str, y_step: int, title_fmt) -> str:
    """One series of columns (no legend: the caption names it); highlighted columns take c-evil, others c-ctrl."""
    left, right, top, bottom, height = 44, 10, 12, 40, 210
    pw, ph = W - left - right, height - top - bottom
    y_hi = max(values) + (-max(values) % y_step or y_step)
    slot = pw / len(values)
    bw = min(16, slot - 4)
    y = lambda v: top + ph * (1 - v / y_hi)
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" class="chart">']
    for t in _ticks(0, y_hi, y_step):
        out.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
    for i, (v, lab) in enumerate(zip(values, labels)):
        cx = left + slot * (i + 0.5)
        cls = "c-evil" if i in highlight else "c-ctrl"
        if v:
            hgt = y(0) - y(v)
            rad = min(4, hgt)
            out.append(
                f'<path class="{cls} fill" d="M{cx - bw / 2:.1f},{y(0):.1f} v-{hgt - rad:.1f} q0,-{rad} {rad},-{rad} '
                f'h{bw - 2 * rad:.1f} q{rad},0 {rad},{rad} v{hgt - rad:.1f} z"><title>{escape(title_fmt(i, v))}</title></path>'
            )
        if i % 2 == 0 or i in highlight:
            out.append(f'<text class="tick" x="{cx:.1f}" y="{top + ph + 16}" text-anchor="middle">{escape(lab)}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    out.append(
        f'<text class="axis-label" x="{left + pw / 2:.0f}" y="{height - 6}" text-anchor="middle">{escape(xlabel)}</text>'
    )
    out.append("</svg>")
    return "".join(out)
