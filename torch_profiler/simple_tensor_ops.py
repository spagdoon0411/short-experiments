"""
Simple tensor ops + PyTorch profiler / CUDA memory snapshot example.

Repeatedly adds a large buffer to itself and runs a softmax, profiling the
steps with torch.profiler and labeling the key regions (add, softmax) via
record_function.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# CUDA memory snapshotting
if device.type == "cuda":
    torch.cuda.memory._record_memory_history(max_entries=100_000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default=None, help="Optional label for this trace run"
    )
    parser.add_argument(
        "--prof_enabled",
        type=bool,
        default=False,
        help="Enable/disable PyTorch profiling",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_id = f"{timestamp}_{args.name}" if args.name else timestamp

    traces_dir = Path(__file__).parent / "traces"
    traces_dir.mkdir(exist_ok=True)

    orig_buf = torch.randn(4096, 16384, device=device) / 100  # batch, feature
    target = torch.clone(orig_buf)

    prof_schedule = schedule(wait=0, warmup=5, active=1, repeat=1)

    prof_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            profile_memory=True,
        )
        if args.prof_enabled
        else nullcontext()
    )

    with prof_ctx as prof, record_function("primary_loop"):
        for _ in range(10):
            with record_function("add"):
                target += orig_buf

        if prof:
            prof.step()

        with record_function("softmax"):
            target = F.softmax(target, dim=-1)  # batch, feature

        if prof:
            prof.step()

    if prof:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        prof.export_chrome_trace(str(traces_dir / f"{trace_id}_trace.json"))

    if device.type == "cuda":
        torch.cuda.memory._dump_snapshot(
            str(traces_dir / f"{trace_id}_cuda_snapshot.pickle")
        )
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
