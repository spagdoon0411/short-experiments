"""
Toy OCR feedforward net + PyTorch profiler / CUDA memory snapshot example.

Trains a tiny MLP on MNIST digits (stand-in for "OCR") and profiles a few
training steps with torch.profiler, labeling the key regions (data load,
forward, backward, optimizer step) via record_function.
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# CUDA memory snapshotting
if device.type == "cuda":
    torch.cuda.memory._record_memory_history(max_entries=100_000)


# toy OCR net: single hidden layer MLP over flattened 28x28 digit images
class OCRNet(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 128, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_dataloader(batch_size: int = 64) -> DataLoader:
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default=None, help="Optional label for this trace run"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_id = f"{timestamp}_{args.name}" if args.name else timestamp

    traces_dir = Path(__file__).parent / "traces"
    traces_dir.mkdir(exist_ok=True)

    model = OCRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_dataloader()

    prof_schedule = schedule(wait=0, warmup=5, active=1, repeat=1)

    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            profile_memory=True,
        ) as prof,
        record_function("primary_loop"),
    ):
        for step, (images, labels) in enumerate(loader):
            if step >= 6:
                break

            with record_function("data_load"):
                images, labels = images.to(device), labels.to(device)

            with record_function("forward"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    prof.export_chrome_trace(str(traces_dir / f"{trace_id}_trace.json"))

    if device.type == "cuda":
        torch.cuda.memory._dump_snapshot(
            str(traces_dir / f"{trace_id}_cuda_snapshot.pickle")
        )
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
