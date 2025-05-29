import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pydantic_settings import BaseSettings
from typing import Optional


class Args(BaseSettings, cli_parse_args=True):
    """Command line arguments for the metrics plotting script."""

    log_file: str
    output_file: Optional[str] = None


def parse_log_line(line):
    # Extract timestamp and metrics
    pattern = r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*? Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
    match = re.search(pattern, line)
    if match:
        timestamp, prompt_tp, gen_tp, running, waiting, cache_usage, hit_rate = (
            match.groups()
        )
        return {
            "timestamp": datetime.strptime(timestamp, "%m-%d %H:%M:%S"),
            "prompt_throughput": float(prompt_tp),
            "generation_throughput": float(gen_tp),
            "running_requests": int(running),
            "waiting_requests": int(waiting),
            "gpu_cache_usage": float(cache_usage),
            "prefix_cache_hit_rate": float(hit_rate),
        }
    return None


def plot_metrics(args: Args):
    # First pass: read the log file to find model information
    model_name = "Unknown Model"
    with open(args.log_file, "r") as f:
        for line in f:
            # Look for model information in the log content
            if "model=" in line:
                # Extract model name from the log line
                model_match = re.search(r"model='([^']+)'", line)
                if model_match:
                    model_name = model_match.group(1)
                break

    # Read and parse the log file for metrics
    data = []
    with open(args.log_file, "r") as f:
        for line in f:
            if "Engine 000:" in line:
                parsed = parse_log_line(line)
                if parsed:
                    data.append(parsed)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f"{model_name} Inference Metrics", fontsize=16)

    # Plot 1: Throughput
    ax1.plot(
        df["timestamp"],
        df["prompt_throughput"],
        label="Prompt Throughput",
        color="blue",
    )
    ax1.plot(
        df["timestamp"],
        df["generation_throughput"],
        label="Generation Throughput",
        color="red",
    )
    ax1.set_ylabel("Tokens/s")
    ax1.set_title("Throughput Over Time")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Request Queue
    ax2.plot(
        df["timestamp"], df["running_requests"], label="Running Requests", color="green"
    )
    ax2.plot(
        df["timestamp"],
        df["waiting_requests"],
        label="Waiting Requests",
        color="orange",
    )
    ax2.set_ylabel("Number of Requests")
    ax2.set_title("Request Queue Status")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Cache Metrics
    ax3.plot(
        df["timestamp"],
        df["gpu_cache_usage"],
        label="GPU KV Cache Usage",
        color="purple",
    )
    ax3.plot(
        df["timestamp"],
        df["prefix_cache_hit_rate"],
        label="Prefix Cache Hit Rate",
        color="brown",
    )
    ax3.set_ylabel("Percentage")
    ax3.set_title("Cache Metrics")
    ax3.legend()
    ax3.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    output_path = args.output_file or f'{model_name.replace("/", "_")}_metrics.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    args = Args()
    plot_metrics(args)


if __name__ == "__main__":
    main()
