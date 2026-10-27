"""Check that report/body.zh.html matches body.en.html structurally and keeps ML terms in English.

uv run python report/check_zh.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
BANNED = ["词元", "令牌", "神经元", "残差流", "激活值", "困惑度", "剂量", "透镜", "提示词", "批次", "贪心"]


def main() -> int:
    en = (HERE / "body.en.html").read_text()
    zh = (HERE / "body.zh.html").read_text()
    readme = (HERE.parent / "README.zh-CN.md").read_text()
    fails = []
    ph = lambda s: Counter(re.findall(r"\[\[\w+\]\]", s))
    if ph(en) != ph(zh):
        fails.append(f"placeholders differ: {(ph(en) - ph(zh)) + (ph(zh) - ph(en))}")
    tags = lambda s: Counter(re.findall(r"<([a-zA-Z][\w-]*)", s))
    if tags(en) != tags(zh):
        fails.append(f"tag counts differ: {(tags(en) - tags(zh)) + (tags(zh) - tags(en))}")
    ids_en, ids_zh = set(re.findall(r'\bid="([^"]+)"', en)), set(re.findall(r'\bid="([^"]+)"', zh))
    if {i + "-zh" for i in ids_en} != ids_zh:
        fails.append(f"ids: {sorted(ids_zh ^ {i + '-zh' for i in ids_en})}")
    bad_href = [h for h in re.findall(r'href="#([^"]+)"', zh) if not h.endswith("-zh")]
    if bad_href:
        fails.append(f"hrefs without -zh: {bad_href}")
    for w in BANNED:
        for name, text in (("body.zh.html", zh), ("README.zh-CN.md", readme)):
            if w in text:
                fails.append(f"banned term {w} in {name}")
    if 'lang="zh-CN"' not in zh.split(">", 1)[0]:
        fails.append("<main> lacks lang=zh-CN")
    blocks = lambda s: re.findall(r"```.*?```", s, re.DOTALL)
    if blocks((HERE.parent / "README.md").read_text()) != blocks(readme):
        fails.append("README code blocks differ")
    print("\n".join(fails) or "body.zh.html and README.zh-CN.md: all checks pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
