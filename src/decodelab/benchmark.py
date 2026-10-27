"""Repeated, synchronized microbenchmarks with raw trials and environment metadata."""

import csv
import json
import platform
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .generation import generate

MODES = ("none", "dynamic", "static")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_benchmark(model, prompt, max_new_tokens=16, warmup=2, repeats=5, seed=7):
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if model.training:
        raise ValueError("benchmark requires model.eval()")
    for _ in range(warmup):
        for mode in MODES:
            generate(model, prompt, max_new_tokens, mode)

    trials = []
    for trial in range(repeats):
        # Rotate order to avoid giving one mode every first/cold measured slot.
        order = MODES[trial % 3 :] + MODES[: trial % 3]
        for mode in order:
            result = generate(model, prompt, max_new_tokens, mode)
            output_tokens = prompt.shape[0] * max_new_tokens
            intervals = [
                (right - left) * 1000
                for left, right in zip(result.token_times_s, result.token_times_s[1:])
            ]
            trials.append(
                {
                    "trial": trial,
                    "mode": mode,
                    "total_ms": result.total_s * 1000,
                    "ttft_ms": result.token_times_s[0] * 1000,
                    "itl_ms": intervals,
                    "output_tokens": output_tokens,
                    "output_tokens_per_s": output_tokens / result.total_s,
                    "cache_used_bytes": result.cache_used_bytes,
                    "cache_allocated_bytes": result.cache_allocated_bytes,
                    "cache_written_tokens_per_sequence": (
                        prompt.shape[1] + max_new_tokens - 1 if mode != "none" else 0
                    ),
                    "cuda_peak_allocated_bytes": result.cuda_peak_allocated_bytes,
                    "cuda_baseline_allocated_bytes": result.cuda_baseline_allocated_bytes,
                }
            )
            # Do not retain a previous generation's output tensor in the next baseline.
            del result

    summary = []
    for mode in MODES:
        rows = [row for row in trials if row["mode"] == mode]
        intervals = [interval for row in rows for interval in row["itl_ms"]]
        peaks = [
            row["cuda_peak_allocated_bytes"]
            for row in rows
            if row["cuda_peak_allocated_bytes"] is not None
        ]
        summary.append(
            {
                "mode": mode,
                "trials": repeats,
                "median_total_ms": statistics.median(row["total_ms"] for row in rows),
                "median_ttft_ms": statistics.median(row["ttft_ms"] for row in rows),
                "median_output_tokens_per_s": statistics.median(
                    row["output_tokens_per_s"] for row in rows
                ),
                "p50_itl_ms": percentile(intervals, 0.50) if intervals else None,
                "p95_itl_ms": percentile(intervals, 0.95) if intervals else None,
                "cache_used_bytes": rows[0]["cache_used_bytes"],
                "cache_allocated_bytes": rows[0]["cache_allocated_bytes"],
                "max_cuda_peak_allocated_bytes": max(peaks) if peaks else None,
            }
        )

    device = prompt.device
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "weights": "seeded random initialization; not trained",
        "model_config": asdict(model.config),
        "request": {
            "batch_size": prompt.shape[0],
            "prompt_tokens": prompt.shape[1],
            "new_tokens": max_new_tokens,
            "prompt_ids": prompt.cpu().tolist(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "os": platform.platform(),
            "cpu": platform.processor(),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu_threads": torch.get_num_threads(),
        },
        "methodology": {
            "warmup_per_mode": warmup,
            "repeats_per_mode": repeats,
            "order": "rotate modes each measured trial",
            "timing": "wall clock; CUDA synchronize before start and after every token",
            "scope": "clone prompt, initialize cache, model calls, greedy selection, append output",
            "ttft": "generation call start to first selected token ready; no network/tokenization",
            "itl": "consecutive selected-token timestamps; batched step timing",
            "memory": "cache tensor storage; CUDA active tensor peak includes model and inputs",
            "limitations": (
                "educational Python path, explicit validation, random weights, no serving SLO"
            ),
        },
        "trials": trials,
        "summary": summary,
    }


def write_report(report, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["summary"][0]))
        writer.writeheader()
        writer.writerows(report["summary"])
    return json_path, csv_path
