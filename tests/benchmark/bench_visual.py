import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "results.json")

with RESULTS_PATH.open() as f:
    data = json.load(f)

rows: dict[tuple[tuple[int, int], str, str, str], dict[str, dict[str, float]]] = {}
SAMPLE_KEYS = [
    ((256, 256), "rgb", "box_blur", "replicate"),
    ((512, 512), "grayscale", "box_blur", "reflect"),
    ((512, 512), "grayscale", "gaussian_blur", "reflect"),
    ((512, 512), "grayscale", "sobel_x", "reflect"),
    ((512, 512), "grayscale", "sharpen", "reflect"),
    ((512, 512), "rgb", "box_blur", "reflect"),
    ((512, 512), "rgb", "gaussian_blur", "reflect"),
    ((512, 512), "rgb", "sobel_x", "reflect"),
    ((512, 512), "rgb", "sharpen", "reflect"),
    ((1024, 1024), "grayscale", "box_blur", "reflect"),
    ((1024, 1024), "rgb", "box_blur", "reflect"),
    ((3840, 2160), "grayscale", "box_blur", "reflect"),
    ((3840, 2160), "rgb", "box_blur", "reflect"),
]

for b in data["benchmarks"]:
    p = b["params"]
    name = b["name"]
    if "my_convolution" in name:
        impl = "custom"
    elif "opencv" in name:
        impl = "opencv"
    else:
        impl = "pillow"

    itype = p["image_type"]
    edge = p["edge_mode_name"]
    kernel = p["kernel_name"]
    mean_ms = b["stats"]["mean"] * 1000
    stddev_ms = b["stats"]["stddev"] * 1000

    key = (tuple(p["image_size"]), itype, kernel, edge)
    rows.setdefault(key, {})[impl] = {"mean": mean_ms, "stddev": stddev_ms}


def completed_rows() -> list[dict[str, Any]]:
    result = []
    for key, values in rows.items():
        if {"custom", "opencv", "pillow"} <= values.keys():
            size, itype, kernel, edge = key
            result.append(
                {
                    "size": size,
                    "itype": itype,
                    "kernel": kernel,
                    "edge": edge,
                    "custom": values["custom"],
                    "opencv": values["opencv"],
                    "pillow": values["pillow"],
                }
            )
    return result


def calc_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def speedup_summary() -> dict[str, dict[str, float]]:
    opencv_speedups = []
    pillow_speedups = []

    for row in completed_rows():
        custom_ms = row["custom"]["mean"]
        opencv_ms = row["opencv"]["mean"]
        pillow_ms = row["pillow"]["mean"]
        opencv_speedups.append(custom_ms / opencv_ms)
        pillow_speedups.append(custom_ms / pillow_ms)

    return {
        "opencv": calc_stats(opencv_speedups),
        "pillow": calc_stats(pillow_speedups),
    }


def fmt_speedup(value: float) -> str:
    return f"~{value:.1f}x"


def fmt_ms(stats: dict[str, float]) -> str:
    precision = 3 if stats["mean"] < 1 else 2
    return f"{stats['mean']:.{precision}f} ± {stats['stddev']:.{precision}f}"


def sample_rows() -> list[dict[str, Any]]:
    result = []
    for size, itype, kernel, edge in SAMPLE_KEYS:
        result.append(
            {
                "size": size,
                "itype": itype,
                "kernel": kernel,
                "edge": edge,
                "values": rows[(size, itype, kernel, edge)],
            }
        )
    return result


def summary_table() -> str:
    summary = speedup_summary()
    labels = {
        "min": "Минимальное ускорение",
        "median": "Медианное ускорение",
        "mean": "Среднее ускорение",
        "max": "Максимальное ускорение",
    }
    lines = [
        "| Метрика | OpenCV быстрее собственной | Pillow быстрее собственной |",
        "|---|---:|---:|",
    ]
    for key, label in labels.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt_speedup(summary["opencv"][key]),
                    fmt_speedup(summary["pillow"][key]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def sample_table() -> str:
    lines = [
        "| Размер | Ядро | Режим | Тип | Собственная реализация, ms | OpenCV, ms | Pillow, ms | OpenCV быстрее | Pillow быстрее |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sample_rows():
        size = row["size"]
        itype = row["itype"]
        kernel = row["kernel"]
        edge = row["edge"]
        values = row["values"]
        custom_ms = values["custom"]["mean"]
        opencv_ms = values["opencv"]["mean"]
        pillow_ms = values["pillow"]["mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{size[0]}x{size[1]}",
                    kernel,
                    edge,
                    itype,
                    fmt_ms(values["custom"]),
                    fmt_ms(values["opencv"]),
                    fmt_ms(values["pillow"]),
                    fmt_speedup(custom_ms / opencv_ms),
                    fmt_speedup(custom_ms / pillow_ms),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def scaling_rows(itype: str) -> list[dict[str, Any]]:
    result = []
    sizes = sorted(
        {
            size
            for size, row_itype, kernel, edge in rows
            if row_itype == itype and kernel == "box_blur" and edge == "reflect"
        },
        key=lambda size: size[0] * size[1],
    )

    for size in sizes:
        values = rows[(size, itype, "box_blur", "reflect")]
        megapixels = size[0] * size[1] / 1_000_000
        result.append(
            {
                "size": size,
                "custom": values["custom"]["mean"] / megapixels,
                "opencv": values["opencv"]["mean"] / megapixels,
                "pillow": values["pillow"]["mean"] / megapixels,
            }
        )
    return result


def scaling_table(itype: str) -> str:
    title = "Grayscale" if itype == "grayscale" else "RGB"
    lines = [
        f"### {title}, box_blur, reflect",
        "",
        "| Размер | Собственная реализация, ms/MP | OpenCV, ms/MP | Pillow, ms/MP |",
        "|---|---:|---:|---:|",
    ]
    for row in scaling_rows(itype):
        size = row["size"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{size[0]}x{size[1]}",
                    f"{row['custom']:.2f}",
                    f"{row['opencv']:.2f}",
                    f"{row['pillow']:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def scaling_tables() -> str:
    return "\n\n".join(
        [
            scaling_table("grayscale"),
            scaling_table("rgb"),
        ]
    )


def write_markdown_tables() -> None:
    output_path = RESULTS_PATH.parent / "benchmark_tables.md"
    output_path.write_text(
        "\n\n".join(
            [
                "# Benchmark tables",
                "## Сводка ускорений",
                summary_table(),
                "## Выборочные результаты",
                sample_table(),
                "## Масштабирование по размеру изображения",
                scaling_tables(),
                "",
            ]
        ),
        encoding="utf-8",
    )


impls = {
    "custom": "Custom",
    "opencv": "OpenCV",
    "pillow": "Pillow",
}
chart_rows = sample_rows()
labels = [
    f"{row['size'][0]}x{row['size'][1]}\n{row['kernel']}\n{row['itype']}, {row['edge']}"
    for row in chart_rows
]
width = 0.25
x = np.arange(len(chart_rows))

fig, ax = plt.subplots(figsize=(17, 7))
for i, (impl, label) in enumerate(impls.items()):
    ys = [row["values"][impl]["mean"] for row in chart_rows]
    ax.bar(x + i * width, ys, width, label=label)

ax.set_yscale("log")
ax.set_xticks(x + width)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_xlabel("Selected benchmark case")
ax.set_ylabel("Execution Time (ms)")
ax.set_title("Comparison on Selected Benchmark Results")
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_PATH.parent / "benchmark_bar.png", dpi=150)
write_markdown_tables()
