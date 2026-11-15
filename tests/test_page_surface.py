"""Small contract tests for the page's first-paint, human-facing numbers."""

from html.parser import HTMLParser
from pathlib import Path
import re


PAGE = Path(__file__).parents[1] / "web" / "index.html"


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self._hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._hidden:
            self._hidden -= 1

    def handle_data(self, data):
        if not self._hidden:
            self.parts.append(data)


def _visible_text() -> str:
    parser = _VisibleText()
    parser.feed(PAGE.read_text())
    return " ".join(" ".join(parser.parts).split())


def _module_text() -> str:
    page = PAGE.read_text()
    return page.split('<script type="module">', 1)[1].split("</script>", 1)[0]


def test_first_paint_is_tier_first_and_has_no_long_decimal_values():
    text = _visible_text()
    assert "382 of 382 loose sorted" not in text
    assert "Junction's brick-sorting station, as a simulator." in text
    assert "99.5 %" in text
    assert "91.3 %" in text
    assert "67.1 %" in text
    assert not re.search(r"(?<![\w.])\d+\.\d{2,}(?![\w.])", text)


def test_null_accuracy_is_not_coerced_to_zero():
    script = _module_text()
    assert "function finiteNumber(value)" in script
    assert "Number(d.accuracy?.combined||0)" not in script
    assert "fmtPct(combined===null?null:combined*100)" in script
