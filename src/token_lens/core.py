"""High-level XAI API."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
from pathlib import Path
import re

from .methods import lime_explain, shap_explain, attention_explain


@dataclass
class Visualization:
    """A token-level explanation."""
    token_scores: List[Tuple[str, float]]
    method: str = "lime"
    label: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        tops = sorted(self.token_scores, key=lambda x: abs(x[1]), reverse=True)[:5]
        return (
            f"Method: {self.method} | Label: {self.label or 'n/a'}\n"
            f"Top drivers: {', '.join(f'{t!r}({s:+.2f})' for t, s in tops)}"
        )

    def show(self, max_width: int = 26):
        """Print terminal bar chart."""
        if not self.token_scores:
            print("(no tokens)")
            return
        mx = max(abs(s) for _, s in self.token_scores) or 1.0
        for tok, s in self.token_scores:
            bar = "#" * int(abs(s) / mx * max_width)
            sign = "+" if s >= 0 else "-"
            print(f"  {tok:<18} {bar:<{max_width}}  {sign}{abs(s):.2f}")

    def to_html(self, path: str):
        parts = []
        mx = max(abs(s) for _, s in self.token_scores) or 1.0
        for tok, s in self.token_scores:
            norm = s / mx
            if norm >= 0:
                bg = f"rgba(124,92,255,{abs(norm):.2f})"
            else:
                bg = f"rgba(255,92,172,{abs(norm):.2f})"
            parts.append(
                f'<span class="tok" style="background:{bg}" '
                f'title="{s:+.3f}">{_escape(tok)}</span>'
            )
        html = f"""<!doctype html><meta charset="utf-8">
<title>Token Lens — {self.method}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:20px;background:#0b0b12;color:#e8e9ee;line-height:2.4}}
h1{{font-size:1.3rem}} .meta{{font-family:ui-monospace,monospace;font-size:0.82rem;color:#a0a3b0;margin-bottom:24px}}
.tok{{display:inline-block;padding:4px 10px;margin:3px;border-radius:5px;font-family:ui-monospace,monospace;cursor:help}}
.legend{{margin-top:30px;font-size:0.8rem;color:#a0a3b0}}
.sw{{display:inline-block;width:18px;height:12px;vertical-align:middle;margin:0 6px;border-radius:3px}}
</style>
<h1>Token importance — {self.method.upper()}</h1>
<div class="meta">label: {self.label or 'n/a'} &middot; top driver: {_top(self.token_scores)}</div>
<div>{''.join(parts)}</div>
<div class="legend">
<span class="sw" style="background:rgba(124,92,255,0.9)"></span>supports prediction
<span class="sw" style="background:rgba(255,92,172,0.9)"></span>opposes prediction
</div>
"""
        Path(path).write_text(html, encoding="utf-8")

    def to_markdown(self) -> str:
        return "| token | score |\n|---|---|\n" + "\n".join(
            f"| `{_escape(t)}` | {s:+.3f} |" for t, s in self.token_scores
        )

    def as_dict(self) -> dict:
        return {"method": self.method, "label": self.label, "tokens": self.token_scores}


def _top(scores):
    if not scores:
        return "n/a"
    t, s = max(scores, key=lambda x: abs(x[1]))
    return f"{t!r} ({s:+.2f})"


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- main API ----

class TokenLens:
    def __init__(self, model_or_predict):
        if callable(model_or_predict):
            self.predict_fn = model_or_predict
            self.model_name = "custom"
        else:
            self.model_name = model_or_predict
            self.predict_fn = self._load_model(model_or_predict)

    @classmethod
    def from_predict_fn(cls, fn, label_names: List[str] | None = None):
        inst = cls(fn)
        inst.label_names = label_names
        return inst

    def _load_model(self, name):
        # Proxy predict: token-overlap-based sentiment-like scoring. Real
        # implementation would use Hugging Face — kept dep-free for baseline use.
        POS = {"good", "great", "love", "amazing", "wonderful", "best", "excellent"}
        NEG = {"bad", "terrible", "hate", "boring", "worst", "awful", "poor"}
        def pred(texts):
            out = []
            for t in texts:
                toks = set(re.findall(r"[a-zA-Z]+", t.lower()))
                s = len(toks & POS) - len(toks & NEG)
                out.append(1 / (1 + math_exp(-s)))
            return out
        return pred

    def explain(self, text: str, method: str = "lime", label: str = "") -> Visualization:
        if method == "lime":
            scores = lime_explain(text, self.predict_fn)
        elif method == "shap":
            scores = shap_explain(text, self.predict_fn)
        elif method == "attention":
            scores = attention_explain(text)
        else:
            raise ValueError(f"Unknown method: {method}")
        return Visualization(token_scores=scores, method=method, label=label)

    def compare(self, text: str, methods=("attention", "lime", "shap")) -> List[Visualization]:
        return [self.explain(text, method=m) for m in methods]


class QALens(TokenLens):
    def explain(self, question: str, context: str, method: str = "lime"):
        combined = f"{question} [SEP] {context}"
        return super().explain(combined, method=method, label="qa")


def explain(model: str | Callable, text: str, method: str = "lime", label: str = "") -> Visualization:
    return TokenLens(model).explain(text, method=method, label=label)


# Minimal math.exp fallback (avoids importing math globally)
def math_exp(x):
    import math
    return math.exp(x)
