"""
Compiled tensor ops + PyTorch profiler / CUDA memory snapshot example.

Same computation as simple_tensor_ops.py (repeated add, then softmax), but
the whole computation is wrapped in a single function and compiled with
torch.compile as one graph, rather than compiling each op as its own segment.
The graph is compiled once before the profiled run so the profiler captures
compiled execution, not compilation.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
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


def compute(target, orig_buf):
    for _ in range(10):
        target = target + orig_buf
    return torch.softmax(target, dim=-1)


compiled_compute = torch.compile(compute)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_id = f"{timestamp}_{args.name}" if args.name else timestamp

    traces_dir = Path(__file__).parent / "traces"
    traces_dir.mkdir(exist_ok=True)

    orig_buf = torch.randn(4096, 16384, device=device) / 100  # batch, feature
    target = torch.clone(orig_buf)

    # Compile the graph before running the profiled loop.
    compiled_compute(target, orig_buf)

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
        with record_function("compute"):
            target = compiled_compute(target, orig_buf)

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
