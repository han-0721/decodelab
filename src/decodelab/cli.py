"""Run with `python -m decodelab --help` or the installed `decodelab` command."""

import argparse

import torch

from .benchmark import MODES, run_benchmark, write_report
from .cache import KVCache
from .generation import generate
from .model import ModelConfig, TinyDecoder


def parser():
    root = argparse.ArgumentParser(description="DecodeLab: inspect forward and exact KV caching")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("demo", "trace", "check", "benchmark"):
        sub = commands.add_parser(command)
        sub.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
        sub.add_argument("--seed", type=int, default=7)
        sub.add_argument("--dim", type=int, default=128)
        sub.add_argument("--heads", type=int, default=4)
        sub.add_argument("--layers", type=int, default=2)
        sub.add_argument("--vocab-size", type=int, default=256)
        sub.add_argument("--context", type=int, default=512)
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--prompt", default="1,2,3,4", help="comma-separated integer token IDs")
        sub.add_argument("--new-tokens", type=int, default=16)
        if command == "demo":
            sub.add_argument("--cache", choices=MODES, default="dynamic")
        if command == "benchmark":
            sub.add_argument(
                "--prompt-len",
                type=int,
                default=64,
                help="length of seeded synthetic prompt; --prompt is unused here",
            )
            sub.add_argument("--batch-size", type=int, default=1)
            sub.add_argument("--warmup", type=int, default=2)
            sub.add_argument("--repeats", type=int, default=5)
            sub.add_argument("--output", default="results/benchmark", help="report path prefix")
    return root


def build_model(args):
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable; use --device cpu or install a CUDA-enabled PyTorch build"
        )
    torch.manual_seed(args.seed)
    config = ModelConfig(args.vocab_size, args.dim, args.heads, args.layers, args.context)
    return TinyDecoder(config).to(device).eval(), torch.device(device)


@torch.inference_mode()
def check(model, prompt, new_tokens):
    reference = model(prompt)
    outputs = {}
    for mode in MODES:
        if mode != "none":
            cache = KVCache(mode, capacity=model.config.max_seq_len)
            # Token-by-token calls test both cached masking and position offsets.
            pieces = [
                model(prompt[:, index : index + 1], cache=cache) for index in range(prompt.shape[1])
            ]
            actual = torch.cat(pieces, dim=1)
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
            error = (actual - reference).abs().max().item()
            print(f"PASS {mode}: max absolute logit error = {error:.3g}")
        outputs[mode] = generate(model, prompt, new_tokens, mode).tokens
    for mode in ("dynamic", "static"):
        if not torch.equal(outputs["none"], outputs[mode]):
            raise AssertionError(f"greedy token mismatch for {mode}")
    print("PASS greedy generation: none == dynamic == static for this fixture")


@torch.inference_mode()
def trace_forward(model, prompt):
    cache = KVCache("dynamic", capacity=model.config.max_seq_len)
    for phase in ("PREFILL", "DECODE"):
        entries = []
        logits = model(prompt, cache=cache, trace=entries)
        print(f"\n{phase}")
        for entry in entries:
            print(f"  {entry['name']:<35} {entry['shape']}")
        prompt = logits[:, -1].argmax(dim=-1, keepdim=True)


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        model, device = build_model(args)
        print(f"DecodeLab | device={device} | seeded UNTRAINED weights | seed={args.seed}")
        if args.new_tokens <= 0:
            raise ValueError("new-tokens must be positive")
        if args.command == "benchmark":
            if args.prompt_len <= 0 or args.batch_size <= 0:
                raise ValueError("prompt-len and batch-size must be positive")
            # A dedicated CPU RNG gives the same prompt across CPU and CUDA runs.
            rng = torch.Generator().manual_seed(args.seed + 1)
            prompt = torch.randint(
                args.vocab_size, (args.batch_size, args.prompt_len), generator=rng
            ).to(device)
            report = run_benchmark(
                model, prompt, args.new_tokens, args.warmup, args.repeats, args.seed
            )
            for row in report["summary"]:
                print(
                    f"{row['mode']:<8} total={row['median_total_ms']:.3f} ms  "
                    f"TTFT={row['median_ttft_ms']:.3f} ms  "
                    f"output={row['median_output_tokens_per_s']:.1f} tok/s"
                )
            paths = write_report(report, args.output)
            print("Saved: " + ", ".join(str(path) for path in paths))
        else:
            try:
                ids = [int(value.strip()) for value in args.prompt.split(",")]
            except ValueError as error:
                raise ValueError(
                    "prompt must contain non-empty comma-separated integer IDs"
                ) from error
            prompt = torch.tensor([ids], dtype=torch.long, device=device)
            if args.command == "trace":
                trace_forward(model, prompt)
            elif args.command == "check":
                check(model, prompt, args.new_tokens)
            else:
                result = generate(model, prompt, args.new_tokens, args.cache)
                print(f"Input token IDs:  {ids}")
                print(f"Output token IDs: {result.tokens[0].tolist()}")
                print(
                    f"TTFT={result.token_times_s[0] * 1000:.3f} ms; "
                    f"total={result.total_s * 1000:.3f} ms"
                )
                print("These IDs demonstrate computation; they are not trained language output.")
        return 0
    except (ValueError, AssertionError) as error:
        root.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
