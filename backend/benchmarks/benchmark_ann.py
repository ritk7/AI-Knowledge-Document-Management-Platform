"""Benchmark: custom HNSW vs ChromaDB vs brute force.

Measures query latency, recall@10, and memory across dataset sizes, plus an
`ef_search` sweep showing the recall/latency tradeoff that defines an ANN index.

Run:
    cd backend && python benchmarks/benchmark_ann.py

Outputs, written next to this script:
    ann_benchmark.png          four-panel comparison chart
    ann_benchmark_results.md   markdown tables for the README
    ann_benchmark_results.json raw numbers

Methodology
-----------
* **Memory is measured in an isolated subprocess per (method, size).** An RSS
  delta taken inside a shared process is dominated by allocator noise and by
  whatever the previous measurement left resident -- the first version of this
  script reported 0.0 MB for a 5,000-vector graph, which is obviously wrong.
  Each measurement now runs in a fresh interpreter: it computes ground truth,
  frees it, forces a collection, samples baseline RSS, then builds only the
  index under test.

* **ChromaDB's memory includes fixed runtime overhead** (~25 MB of client and
  ONNX runtime init) that does not scale with dataset size. It is reported as
  measured rather than subtracted out, because it is a real cost of using
  Chroma -- but it means the *slope* across sizes, not the absolute value, is
  the meaningful comparison.

* **Vectors come from a Gaussian mixture with deliberately overlapping
  clusters.** Well-separated clusters make every ANN method score a perfect
  recall@10 and hide the tradeoff entirely. Real embeddings are clustered but
  with substantial overlap, so cluster noise is tuned to keep nearest
  neighbours genuinely ambiguous.

* ChromaDB uses HNSW internally (the C++ `hnswlib`). This is therefore largely
  *the same algorithm in Python vs in C++*, which is the honest framing: the
  latency gap is an implementation result, not an algorithmic one.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DIM = 384  # matches all-MiniLM-L6-v2
SIZES = [100, 1000, 5000]
NUM_QUERIES = 100
K = 10
SEED = 42
EF_SWEEP = [10, 16, 32, 64, 128, 256]
EF_SWEEP_SIZE = 5000

METHODS = ["brute", "hnsw", "chroma"]
METHOD_LABELS = {
    "brute": "Brute force (exact)",
    "hnsw": "Custom HNSW (this repo)",
    "chroma": "ChromaDB (hnswlib, C++)",
}
COLORS = {
    "brute": "#94a3b8",
    "hnsw": "#4f46e5",
    "chroma": "#0ea5e9",
}

OUT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


def make_dataset(n: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Clustered vectors with overlapping clusters.

    Cluster noise (0.9) is large relative to the spread of cluster centres
    (1.0), so a query's true nearest neighbours are genuinely contested between
    nearby clusters. With well-separated clusters every method trivially scores
    recall 1.0 and the benchmark measures nothing.
    """
    rng = np.random.default_rng(seed)
    num_clusters = max(2, n // 100)
    centers = rng.normal(scale=1.0, size=(num_clusters, dim))

    assignments = rng.integers(0, num_clusters, size=n)
    data = centers[assignments] + rng.normal(scale=0.9, size=(n, dim))

    query_clusters = rng.integers(0, num_clusters, size=NUM_QUERIES)
    queries = centers[query_clusters] + rng.normal(scale=0.9, size=(NUM_QUERIES, dim))
    return data.astype(np.float32), queries.astype(np.float32)


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


# ----------------------------------------------------------------------
# Worker: one isolated measurement
# ----------------------------------------------------------------------


def run_worker(method: str, n: int, ef: int | None, sweep: bool = False) -> dict:
    """Measure one (method, size) in this fresh interpreter. Prints JSON.

    With ``sweep`` set (HNSW only), the same graph is additionally queried at
    every ``ef_search`` in EF_SWEEP. ``ef_search`` is a *query-time* beam width,
    so it needs no rebuild -- reusing one graph makes the sweep a controlled
    comparison (identical edges throughout) and avoids six redundant builds.
    """
    from app.hnsw import BruteForceIndex, HNSWIndex

    data, queries = make_dataset(n, DIM, SEED)
    labels = [str(i) for i in range(n)]

    # Memory is sampled around the index under test and nothing else.
    #
    # An earlier version built the brute-force ground truth first and freed it
    # before sampling the baseline. That silently broke the measurement: freeing
    # a large array returns its pages to CPython's allocator arena rather than
    # to the OS, so the index under test allocated into that reserved arena and
    # RSS never grew. It reported 0.0 MB for a 5,000-node graph. Ground truth is
    # therefore computed *after* the memory sample instead.
    gc.collect()
    baseline = rss_mb()

    # ---- Build ----
    build_start = time.perf_counter()
    if method == "brute":
        index = BruteForceIndex(DIM)
        index.add_batch(data, labels)
        search = lambda q: index.search(q, K)  # noqa: E731
    elif method == "hnsw":
        index = HNSWIndex(
            dim=DIM,
            M=16,
            ef_construction=200,
            ef_search=ef or 64,
            seed=SEED,
        )
        index.add_batch(data, labels)
        search = lambda q: index.search(q, K)  # noqa: E731
    elif method == "chroma":
        import chromadb

        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            name=f"bench_{n}", metadata={"hnsw:space": "cosine"}
        )
        collection.add(ids=labels, embeddings=data.tolist())
        index = collection

        def search(q):
            result = collection.query(query_embeddings=[q.tolist()], n_results=K)
            return [(cid, 0.0) for cid in result["ids"][0]]
    else:
        raise ValueError(f"unknown method: {method}")

    build_ms = (time.perf_counter() - build_start) * 1000
    gc.collect()

    # Memory: deterministic accounting where the structure is introspectable,
    # RSS delta only where it is not.
    #
    # RSS delta proved unusable for the in-process Python indexes. numpy
    # temporaries allocated while generating the dataset are freed back into
    # CPython's arena rather than to the OS, so the index under test allocates
    # into that reserved space and RSS does not grow -- it reported 0.0 MB for a
    # 5,000-node graph while correctly reporting 3.42 MB for a 1,000-node one.
    # Counting the bytes directly is exact, reproducible, and not subject to
    # allocator behaviour. ChromaDB allocates in C++ behind an opaque API, so it
    # keeps the RSS delta (which includes ~70 MB of fixed client/ONNX init that
    # does not scale with dataset size).
    if method == "chroma":
        memory_mb = max(0.0, rss_mb() - baseline)
        memory_method = "rss"
    else:
        memory_mb = (
            index.memory_bytes()
            if method == "brute"
            else index.stats()["vector_bytes"] + index.stats()["graph_bytes_approx"]
        ) / (1024 * 1024)
        memory_method = "exact"

    # ---- Ground truth (after the memory sample, see note above) ----
    if method == "brute":
        # The index under test is already exact -- reuse it rather than
        # allocating a second copy.
        ground_truth = [{label for label, _ in index.search(q, K)} for q in queries]
    else:
        truth_index = BruteForceIndex(DIM)
        truth_index.add_batch(data, labels)
        ground_truth = [
            {label for label, _ in truth_index.search(q, K)} for q in queries
        ]

    # ---- Query ----
    timings, recalls = [], []
    for query, truth in zip(queries, ground_truth):
        start = time.perf_counter()
        result = search(query)
        timings.append((time.perf_counter() - start) * 1000)
        retrieved = {label for label, _ in result}
        recalls.append(len(retrieved & truth) / len(truth) if truth else 0.0)

    row = {
        "method": method,
        "label": METHOD_LABELS[method],
        "size": n,
        "ef_search": ef,
        "build_ms": round(build_ms, 1),
        "query_mean_ms": round(float(np.mean(timings)), 3),
        "query_p95_ms": round(float(np.percentile(timings, 95)), 3),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "memory_mb": round(memory_mb, 2),
        "memory_method": memory_method,
    }

    if method == "hnsw":
        stats = index.stats()
        row["avg_degree_layer0"] = round(stats["avg_degree_layer0"], 1)
        row["top_level"] = stats["top_level"]
        row["index_bytes"] = stats["vector_bytes"] + stats["graph_bytes_approx"]

    if not sweep:
        return row

    # ---- ef_search sweep against the graph we already built ----
    sweep_rows = []
    for ef_value in EF_SWEEP:
        ef_timings, ef_recalls = [], []
        for query, truth in zip(queries, ground_truth):
            start = time.perf_counter()
            result = index.search(query, K, ef=ef_value)
            ef_timings.append((time.perf_counter() - start) * 1000)
            retrieved = {label for label, _ in result}
            ef_recalls.append(len(retrieved & truth) / len(truth) if truth else 0.0)
        sweep_rows.append(
            {
                "ef_search": ef_value,
                "query_mean_ms": round(float(np.mean(ef_timings)), 3),
                "recall_at_10": round(float(np.mean(ef_recalls)), 4),
            }
        )

    return {"main": row, "sweep": sweep_rows}


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def measure(method: str, n: int, ef: int | None = None, sweep: bool = False) -> dict:
    """Spawn a fresh interpreter for one measurement."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", method, str(n)]
    if ef is not None:
        cmd += ["--ef", str(ef)]
    if sweep:
        cmd += ["--sweep"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker {method}/{n} failed:\n{completed.stdout}\n{completed.stderr}"
        )
    # The worker prints its JSON on the last non-empty line; libraries such as
    # Chroma write banners to stdout that we have to skip past.
    line = [ln for ln in completed.stdout.strip().splitlines() if ln.strip()][-1]
    return json.loads(line)


def write_chart(rows: list[dict], sweep: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    fig.suptitle(
        f"ANN index comparison — {DIM}-dim vectors, k={K}, {NUM_QUERIES} queries "
        f"(isolated subprocess per measurement)",
        fontsize=13,
        fontweight="bold",
    )

    def series(method: str, field: str):
        points = [r for r in rows if r["method"] == method]
        return [p["size"] for p in points], [p[field] for p in points]

    # Panel 1 — query latency
    ax = axes[0]
    for method in METHODS:
        x, y = series(method, "query_mean_ms")
        ax.plot(
            x, y, marker="o", label=METHOD_LABELS[method], color=COLORS[method],
            linewidth=2,
        )
    ax.set(
        xlabel="Dataset size (vectors)",
        ylabel="Mean query latency (ms)",
        title="Query latency — lower is better",
        xscale="log",
        yscale="log",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Panel 2 — recall@10
    ax = axes[1]
    width = 0.26
    positions = np.arange(len(SIZES))
    for offset, method in enumerate(METHODS):
        _, y = series(method, "recall_at_10")
        ax.bar(
            positions + (offset - 1) * width,
            y,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set(
        xlabel="Dataset size (vectors)",
        ylabel="recall@10",
        title="Recall@10 vs exact search",
        ylim=(0, 1.15),
    )
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower right")

    # Panel 3 — memory
    ax = axes[2]
    for method in METHODS:
        x, y = series(method, "memory_mb")
        ax.plot(
            x, y, marker="s", label=METHOD_LABELS[method], color=COLORS[method],
            linewidth=2,
        )
    ax.set(
        xlabel="Dataset size (vectors)",
        ylabel="Index memory (MB)",
        title="Memory — exact for Python; Chroma is RSS (~70 MB fixed init)",
        xscale="log",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Panel 4 — ef_search tradeoff
    ax = axes[3]
    ef_values = [s["ef_search"] for s in sweep]
    latencies = [s["query_mean_ms"] for s in sweep]
    recalls = [s["recall_at_10"] for s in sweep]

    ax.plot(latencies, recalls, marker="o", color="#4f46e5", linewidth=2)
    for ef, lat, rec in zip(ef_values, latencies, recalls):
        ax.annotate(
            f"ef={ef}",
            (lat, rec),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=7.5,
            color="#475569",
        )
    ax.set(
        xlabel="Mean query latency (ms)",
        ylabel="recall@10",
        title=f"Custom HNSW: recall/latency knob (n={EF_SWEEP_SIZE})",
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ann_benchmark.png", dpi=150, bbox_inches="tight")
    print(f"\nChart  -> {OUT_DIR / 'ann_benchmark.png'}")


def write_markdown(rows: list[dict], sweep: list[dict]) -> None:
    main = [
        "| Dataset | Method | Build (ms) | Query mean (ms) | Query p95 (ms) | recall@10 | Memory (MB) |",
        "|--------:|--------|-----------:|----------------:|---------------:|----------:|------------:|",
    ]
    for size in SIZES:
        for row in [r for r in rows if r["size"] == size]:
            suffix = "" if row["memory_method"] == "exact" else " ᴿ"
            main.append(
                f"| {row['size']} | {row['label']} | {row['build_ms']} | "
                f"{row['query_mean_ms']} | {row['query_p95_ms']} | "
                f"{row['recall_at_10']} | {row['memory_mb']}{suffix} |"
            )
    main += [
        "",
        "Memory is counted exactly (vectors + graph edges) for the Python "
        "indexes. ᴿ = RSS delta, used only for ChromaDB, whose allocation "
        "happens in C++ behind an opaque API; it includes ~70 MB of fixed "
        "client init that does not scale with dataset size.",
    ]

    sweep_lines = [
        "",
        f"**`ef_search` sweep — custom HNSW, n={EF_SWEEP_SIZE}**",
        "",
        "| ef_search | Query mean (ms) | recall@10 |",
        "|----------:|----------------:|----------:|",
    ]
    for row in sweep:
        sweep_lines.append(
            f"| {row['ef_search']} | {row['query_mean_ms']} | {row['recall_at_10']} |"
        )

    text = "\n".join(main + sweep_lines)
    (OUT_DIR / "ann_benchmark_results.md").write_text(text + "\n")
    (OUT_DIR / "ann_benchmark_results.json").write_text(
        json.dumps({"main": rows, "ef_sweep": sweep}, indent=2)
    )
    print(f"Tables -> {OUT_DIR / 'ann_benchmark_results.md'}\n")
    print(text)


def main() -> None:
    rows: list[dict] = []
    sweep: list[dict] = []

    for size in SIZES:
        print(f"\n{'=' * 64}\nDataset: {size} vectors x {DIM} dims\n{'=' * 64}")
        for method in METHODS:
            # Fold the ef sweep into the HNSW run at the sweep size, so the
            # 5,000-node graph is built once rather than seven times.
            want_sweep = method == "hnsw" and size == EF_SWEEP_SIZE
            result = measure(method, size, sweep=want_sweep)

            if want_sweep:
                row, sweep = result["main"], result["sweep"]
            else:
                row = result

            rows.append(row)
            print(
                f"  {METHOD_LABELS[method]:<26} "
                f"{row['query_mean_ms']:>7.3f} ms/query   "
                f"recall {row['recall_at_10']:.3f}   "
                f"{row['memory_mb']:>6.2f} MB"
            )

    print(f"\n{'=' * 64}\nef_search sweep (custom HNSW, n={EF_SWEEP_SIZE})\n{'=' * 64}")
    for row in sweep:
        print(
            f"  ef={row['ef_search']:<4} {row['query_mean_ms']:>7.3f} ms/query   "
            f"recall {row['recall_at_10']:.3f}"
        )

    write_chart(rows, sweep)
    write_markdown(rows, sweep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("METHOD", "SIZE"))
    parser.add_argument("--ef", type=int, default=None)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.worker:
        method, size = args.worker
        print(json.dumps(run_worker(method, int(size), args.ef, sweep=args.sweep)))
    else:
        main()
