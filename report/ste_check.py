"""Rough ASD-STE100 check of the report prose: sentence length, passive voice, contractions, words to avoid.

    uv run python report/ste_check.py report/body.en.html

It checks rules that a script can check. It does not check words against the STE dictionary.
"""

import re
import sys
from html.parser import HTMLParser

PROSE_TAGS = {"p", "li", "dd", "figcaption", "td", "h1", "h2", "h3"}
AVOID = {
    "about": "approximately / of / on (STE: ABOUT is approved only as 'concerning')",
    "utilize": "use",
    "approx": "approximately",
    "e.g.": "for example",
    "i.e.": "that is",
    "however": "but",
    "therefore": "thus",
    "very": None,
    "seems": "possible / shows",
    "seem": None,
    "although": "but",
    "via": "through / with",
    "found out": "find",
    "figure out": "find",
    "lots": "many",
    "get rid": "remove",
    "basically": None,
    "actually": None,
}
PASSIVE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+(\w+ed|made|done|given|shown|found|written|seen|known|taken|put|set)\b",
    re.IGNORECASE,
)
CONTRACTION = re.compile(r"\b\w+(n't|'re|'ll|'ve|'d)\b", re.IGNORECASE)


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.skip = 0
        self.buf: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script", "svg", "code", "table") and tag != "table":
            self.skip += 1
        if tag in PROSE_TAGS:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in ("style", "script", "svg", "code"):
            self.skip -= 1
            if tag == "code":
                self.buf.append("CODE")
        if tag in PROSE_TAGS:
            self.depth -= 1
            if self.depth == 0 and self.buf:
                self.blocks.append(" ".join("".join(self.buf).split()))
                self.buf = []

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.buf.append(data)


def main(path: str):
    t = Text()
    with open(path) as f:
        t.feed(f.read())
    issues = 0
    n_sent = 0
    for block in t.blocks:
        block = re.sub(r"\[\[\w+\]\]", "X", block)
        for s in re.split(r"(?<=[.!?])\s+", block):
            words = [w for w in s.split() if re.search(r"\w", w)]
            if len(words) < 3:
                continue
            n_sent += 1
            flags = []
            if len(words) > 25:
                flags.append(f"{len(words)} words")
            if m := PASSIVE.search(s):
                flags.append(f"passive? '{m.group(0)}'")
            if m := CONTRACTION.search(s.replace("’", "'")):
                flags.append(f"contraction '{m.group(0)}'")
            low = f" {s.lower()} "
            for w, alt in AVOID.items():
                if re.search(rf"\b{re.escape(w)}\b", low) and not re.search(r"[“\"'].*" + re.escape(w), low):
                    flags.append(f"avoid '{w}'" + (f" -> {alt}" if alt else ""))
            if flags:
                issues += 1
                print(f"- {'; '.join(flags)}\n    {s[:160]}")
    print(f"\n{n_sent} sentences checked, {issues} flagged")


if __name__ == "__main__":
    main(sys.argv[1])
