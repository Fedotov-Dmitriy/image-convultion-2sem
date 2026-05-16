import json
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import DefaultDict

with open(sys.argv[1] if len(sys.argv) > 1 else "results.json") as f:
    data = json.load(f)

TimesDict = DefaultDict[
    str, DefaultDict[str, DefaultDict[str, DefaultDict[str, list[float]]]]
]

times: TimesDict = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
)

for b in data["benchmarks"]:
    p = b["params"]
    name = b["name"]
    if "my_convolution" in name:
        impl = "custom"
    elif "opencv" in name:
        impl = "opencv"
    else:
        impl = "pillow"

    size = f"{p['image_size'][0]}x{p['image_size'][1]}"
    itype = p["image_type"]
    edge = p["edge_mode_name"]
    times[itype][edge][impl][size].append(b["stats"]["mean"] * 1000)

sizes = ["128x128", "256x256", "512x512", "1024x1024"]
impls = ["custom", "opencv", "pillow"]
itypes = ["grayscale", "rgb"]
edges = ["reflect", "replicate"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Comparison of Implementation Performance")

width = 0.25
x = np.arange(len(sizes))

for row, edge in enumerate(edges):
    for col, itype in enumerate(itypes):
        ax = axes[row][col]
        for i, impl in enumerate(impls):
            ys = [np.mean(times[itype][edge][impl][s]) for s in sizes]
            ax.bar(x + i * width, ys, width, label=impl)

        ax.set_yscale("log")
        ax.set_xticks(x + width)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("Image Size")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(f"color_mode = {itype} | edge = {edge}")
        ax.legend()

plt.tight_layout()
plt.savefig("benchmark_bar.png", dpi=150)
