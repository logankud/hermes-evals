"""Matching helpers shared by evaluators: numbers in text, and tool calls against expected specs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall

# A number as people write it: optional sign and $, thousands separators, decimals, a k/M/B suffix, and %.
# Digits glued to a letter (Q2, v3) are not numbers.
_NUMBER = re.compile(
    r"(?<![\w.])(-?)(\$?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?([kKmMbB])(?![A-Za-z]))?(\s?%)?"
)
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9}


@dataclass(frozen=True, slots=True)
class Figure:
    value: float
    plain: bool  # a bare integer like "3": no $, decimals, suffix or %


def figures(text: str) -> list[Figure]:
    """Every number in the text, with whether it was written as a plain integer."""
    out = []
    for sign, dollar, whole, frac, suffix, percent in _NUMBER.findall(text or ""):
        value = float(whole.replace(",", "") + (frac or "")) * _SCALE.get(suffix.lower(), 1.0)
        out.append(Figure(-value if sign else value, plain=not (dollar or frac or suffix or percent)))
    return out


def numbers(text: str) -> list[float]:
    """Every number in the text, as floats: '$48.2k' -> 48200.0, '1,842' -> 1842.0, '4.8%' -> 4.8."""
    return [f.value for f in figures(text)]


def as_number(value: Any) -> float | None:
    """The value as a number if it is one (42, '42', '$1,842', '3.5%'), otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    found = numbers(str(value))
    text = re.sub(r"[\s$%,]", "", str(value))
    return found[0] if len(found) == 1 and re.fullmatch(r"-?\d+(\.\d+)?[kKmMbB]?", text) else None


def close(a: float, b: float, rel_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """An expected tool call: a regex on the tool name, and optionally a regex on its arguments."""

    name: str
    args: str | None = None

    def matches(self, call: ToolCall) -> bool:
        return bool(re.search(self.name, call.name)) and (
            self.args is None or bool(re.search(self.args, call.arguments or "", re.IGNORECASE)))


def tool_specs(expected: Any) -> list[ToolSpec]:
    """The reference trajectory from a case: a list of tool names, or of {"name": ..., "args": ...}."""
    if expected is None:
        return []
    items = expected if isinstance(expected, list) else [expected]
    return [ToolSpec(item) if isinstance(item, str) else ToolSpec(item["name"], item.get("args")) for item in items]


_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_DATES = re.compile(
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?"  # June 30, Jun 30th, June 30, 2026
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:\s+\d{{4}})?"  # 30 June, 30th June 2026
    r"|\b\d{4}-\d{2}-\d{2}\b"  # 2026-04-01
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"  # 4/1, 4/1/2026
    r"|(?<![\d$,.])(?:19|20)\d{2}(?![\d%]|[.,]\d)",  # a bare year; "$2026", "2,026" and "2026.5" stay amounts
    re.IGNORECASE,
)


def without_dates(text: str) -> str:
    """The text with dates and bare years removed, so they aren't mistaken for figures."""
    return _DATES.sub(" ", text or "")
