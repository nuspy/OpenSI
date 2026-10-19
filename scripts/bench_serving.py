"""Serving-endpoint benchmark probe (Milestone 1 completion).

Probes an OpenAI-compatible endpoint (llama-server or similar):

- resolves the served model id via GET {base}/models
- sends the eval-task goals from configs/prototype.yaml as prompts,
  three repetitions each, and times them
- reports latency per request, tokens/second estimates, and whether
  the served model matches the configured model id

Writes machine-readable JSON to stdout (captured by test.py into
results.txt). Requires only the `requests` package; runs on the CPU
side, so it never touches GPU memory directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "configs" / "prototype.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover_server(base_url: str) -> dict | None:
    """Return the /v1/models payload for a reachable server."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def bench(base_url: str, goals: list[str], reps: int = 3) -> dict:
    """Send each goal as a chat request, time each call."""
    samples: list[dict] = []
    for goal in goals:
        for rep in range(reps):
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": goal}],
                        "max_tokens": 64,
                        "temperature": 0.0,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                completion = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}) or {}
                elapsed = time.perf_counter() - t0
                samples.append({
                    "goal": goal,
                    "rep": rep,
                    "ok": True,
                    "seconds": round(elapsed, 4),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "answer_excerpt": str(completion)[:80],
                })
            except Exception as exc:  # noqa: BLE001
                samples.append({
                    "goal": goal, "rep": rep, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    ok = [s for s in samples if s["ok"]]
    total_seconds = sum(s["seconds"] for s in ok)
    total_tokens = sum(s.get("completion_tokens") or 0 for s in ok)
    return {
        "samples": samples,
        "summary": {
            "requests_ok": len(ok),
            "requests_failed": len(samples) - len(ok),
            "total_seconds": round(total_seconds, 3),
            "total_completion_tokens": total_tokens,
            "tokens_per_second": (
                round(total_tokens / total_seconds, 1)
                if total_seconds > 0 else 0.0
            ),
        },
    }


def main() -> None:
    cfg = load_config()
    base = cfg.get("openai_base_url", "http://127.0.0.1:8080/v1")
    models_payload = discover_server(base)
    goals = [t["goal"] for t in cfg.get("eval_tasks", [])]
    bench_result = bench(base, goals) if models_payload else None

    report = {
        "endpoint": base,
        "server_reachable": models_payload is not None,
        "served_models": (
            [m.get("id") for m in models_payload.get("data", [])]
            if models_payload else []
        ),
        "expected_model": cfg.get("model"),
        "benchmark": bench_result,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
