import json

import pytest
import torch

from decodelab.benchmark import run_benchmark, write_report
from decodelab.cli import main
from decodelab.model import ModelConfig, TinyDecoder


def test_benchmark_records_real_trials_and_batch_token_counts(tmp_path):
    torch.manual_seed(3)
    model = TinyDecoder(
        ModelConfig(vocab_size=16, dim=16, n_heads=2, n_layers=1, max_seq_len=16)
    ).eval()
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    report = run_benchmark(model, prompt, max_new_tokens=4, warmup=0, repeats=2, seed=3)
    assert len(report["trials"]) == 6
    assert {row["mode"] for row in report["summary"]} == {"none", "dynamic", "static"}
    for row in report["trials"]:
        assert row["output_tokens"] == 8  # 2 sequences x 4 newly selected tokens
        assert len(row["itl_ms"]) == 3
        assert row["ttft_ms"] > 0
        assert row["total_ms"] >= row["ttft_ms"]
        assert row["output_tokens_per_s"] == pytest.approx(8000 / row["total_ms"])
        if row["mode"] != "none":
            assert row["cache_written_tokens_per_sequence"] == 6
        assert row["cuda_peak_allocated_bytes"] is None
    prefix = tmp_path / "nested" / "trial"
    json_path, csv_path = write_report(report, prefix)
    assert json.loads(json_path.read_text(encoding="utf-8"))["seed"] == 3
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 4


def test_benchmark_rejects_zero_repeats():
    model = TinyDecoder(ModelConfig(dim=8, n_heads=1, n_layers=1))
    with pytest.raises(ValueError, match="repeats"):
        run_benchmark(model, torch.tensor([[1]]), repeats=0)


def test_trace_prints_prefill_and_cached_decode_shapes(capsys):
    assert main(["trace", "--dim", "16", "--heads", "2", "--layers", "1", "--prompt", "1,2,3"]) == 0
    output = capsys.readouterr().out
    assert "PREFILL" in output
    assert "DECODE" in output
    assert "[1, 3, 16]" in output
    assert "[1, 2, 4, 8]" in output  # cached K has prompt + one new token


@pytest.mark.parametrize(
    "arguments",
    [
        ["demo", "--new-tokens", "0"],
        ["demo", "--prompt", ""],
        ["benchmark", "--repeats", "0"],
        ["demo", "--dim", "15", "--heads", "2"],
        ["demo", "--device", "cuda"],  # CPU-only test environment
    ],
)
def test_invalid_cli_inputs_return_a_clear_error(arguments, capsys):
    if "cuda" in arguments and torch.cuda.is_available():
        pytest.skip("CUDA is available on this runner")
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_check_validates_cache_logits_and_generation(capsys):
    assert main(["check", "--dim", "16", "--layers", "1", "--heads", "2", "--new-tokens", "3"]) == 0
    output = capsys.readouterr().out
    assert "PASS" in output
    assert "dynamic" in output and "static" in output
