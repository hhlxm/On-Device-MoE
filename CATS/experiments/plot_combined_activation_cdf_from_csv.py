import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def _parse_float(value: str) -> float:
    value = value.strip().lower()
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def plot_combined_activation_cdf_from_csv(csv_path: str, output_path: str = None):
    bin_right = []
    cdf_percentage = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"bin_left", "bin_right", "cdf_percentage"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_columns)}")

        for row in reader:
            left = _parse_float(row["bin_left"])
            right = _parse_float(row["bin_right"])
            if left < 0 or not np.isfinite(left) or not np.isfinite(right):
                continue
            bin_right.append(right)
            cdf_percentage.append(_parse_float(row["cdf_percentage"]))

    if not bin_right:
        raise ValueError(f"No finite non-negative bins found in {csv_path}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "combined_post_activation_abs_cdf_from_csv.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(
        bin_right,
        cdf_percentage,
        color="#227CF6",
        linewidth=2,
    )
    ax.set_xlabel("|X|")
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(left=0, right=bin_right[-1])
    ax.set_ylim(bottom=0, top=100)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("Combined Post-Activation Absolute CDF")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Reconstruct the combined activation CDF figure from CSV.")
    parser.add_argument("csv_path", help="Path to combined_histogram.csv.")
    parser.add_argument("--output-path", default=None, help="Output figure path. Defaults next to the CSV.")
    args = parser.parse_args()

    output_path = plot_combined_activation_cdf_from_csv(args.csv_path, args.output_path)
    print(output_path)


if __name__ == "__main__":
    main()
