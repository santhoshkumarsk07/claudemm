"""Optional local LLM through Ollama (free and open source: https://ollama.com).

Install Ollama, then `ollama pull llama3.1:8b` (or qwen2.5:7b, mistral). Everything runs on
your machine; no API key, no cost.

The LLM is an ADVISOR only. Its JSON answer can say "agree", "caution" or "veto" about a
proposed new entry and give commentary. Code decides what that means: a veto can only
cancel a new entry. The LLM never creates orders, sizes positions or touches risk limits.

LLMs are not used in historical backtests: a model trained on internet text from after the
test period already "knows" what happened, which is look-ahead bias.
"""
from __future__ import annotations

import json
import logging

import requests

log = logging.getLogger("quant")

SYSTEM = """You review proposed PAPER trades (virtual money) for Indian stocks. You get numbers computed by code:
market regime, indicators, a machine-learning probability, and the risk manager's sizing.
Answer ONLY with JSON: {"stance": "agree"|"caution"|"veto", "commentary": "<2 sentences citing the numbers>"}.
Use "veto" only for a concrete reason visible in the data (e.g. trend and ML disagree, extreme volatility).
You cannot change sizes, stops or rules. Do not invent news or prices you were not given."""


class Ollama:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.enabled, self.host, self.model, self.timeout = c["enabled"], c["host"].rstrip("/"), c["model"], c["timeout_seconds"]
        self._ok: bool | None = None

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._ok is None:
            try:
                r = requests.get(f"{self.host}/api/tags", timeout=3)
                names = [m.get("name", "") for m in r.json().get("models", [])]
                self._ok = r.ok and any(n.split(":")[0] == self.model.split(":")[0] for n in names)
                if r.ok and not self._ok:
                    log.warning("Ollama is running but model %s is not pulled. Run: ollama pull %s", self.model, self.model)
            except requests.RequestException:
                self._ok = False
        return self._ok

    def review(self, proposal: dict) -> dict:
        """Return {"stance", "commentary", "source"}; never raises."""
        if not self.available():
            return {"stance": "unavailable", "commentary": "Local LLM not running; rules and ML decided alone.", "source": "none"}
        try:
            r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
                "model": self.model, "stream": False, "format": "json", "options": {"temperature": 0, "seed": 7},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": json.dumps(proposal, default=str)}]})
            r.raise_for_status()
            out = json.loads(r.json()["message"]["content"])
            stance = str(out.get("stance", "")).lower()
            if stance not in ("agree", "caution", "veto"):
                stance = "caution"
            return {"stance": stance, "commentary": str(out.get("commentary", ""))[:500], "source": f"ollama:{self.model}"}
        except Exception as e:  # malformed JSON, timeouts: fall back to no opinion
            return {"stance": "unavailable", "commentary": f"LLM error ({type(e).__name__}); ignored.", "source": "error"}
