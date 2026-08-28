"""Retrieval evaluation: precision@k / recall@k across pipeline configurations.

Run:
    cd backend && python eval/run_eval.py

Outputs, written next to this script:
    eval_results.md    markdown tables for the README
    eval_results.json  raw per-question numbers

What is being measured
----------------------
Only *retrieval* -- no LLM call is made. Answer quality is downstream of
retrieval: if the right chunk never makes it into the context window, no
prompt can recover it. Measuring the retriever in isolation also makes the
eval free and deterministic.

Ground truth
------------
Labelling chunk IDs by hand would break the moment CHUNK_SIZE changes. Instead
each question carries *evidence phrases*: a chunk is relevant if it belongs to
the labelled document and contains at least one phrase. This is a weak-labelling
scheme, and it is worth being explicit about its bias -- it favours retrievers
that match on literal text. In practice the phrases are the specific facts that
answer the question, so a chunk containing one genuinely is a correct source,
but it does mean these numbers should be read as relative comparisons between
configurations rather than as absolute quality scores.

Metrics
-------
* **precision@k** -- of the k chunks returned, the fraction that are relevant.
  Capped by k/|relevant|: with 1 relevant chunk and k=5, the ceiling is 0.2, so
  compare configurations against each other rather than against 1.0.
* **recall@k** -- of all relevant chunks, the fraction retrieved.
* **hit@k** -- did at least one relevant chunk appear? The metric that most
  directly predicts whether the LLM can answer at all.
* **MRR** -- 1/rank of the first relevant chunk. Sensitive to *ordering*, which
  precision and recall are blind to; this is where re-ranking shows up.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bm25_index import refresh_bm25_index  # noqa: E402
from app.config import settings  # noqa: E402
from app.document_processor import process_file  # noqa: E402
from app.embeddings import embed_texts  # noqa: E402
from app.retrieval import retrieve  # noqa: E402
from app.vector_store import add_chunks, get_all_chunks, reset_collection  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
K_VALUES = [1, 3, 5]

CONFIGS = [
    {"name": "Vector only", "alpha": 1.0, "use_reranker": False},
    {"name": "BM25 only", "alpha": 0.0, "use_reranker": False},
    {"name": "Hybrid (a=0.6)", "alpha": 0.6, "use_reranker": False},
    {"name": "Hybrid + rerank", "alpha": 0.6, "use_reranker": True},
]


def normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so phrase matching survives chunking."""
    return re.sub(r"\s+", " ", text).strip().lower()


def index_corpus() -> None:
    """Wipe the store and index the eval corpus from scratch."""
    print("Re-indexing eval corpus...")
    reset_collection()

    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks, num_pages = process_file(path)
        embeddings = embed_texts([c.text for c in chunks])
        # Use the filename as the document id so eval labels can refer to it.
        add_chunks(path.name, path.name, chunks, embeddings, num_pages)
        print(f"  {path.name}: {len(chunks)} chunks")

    total = refresh_bm25_index()
    print(f"  BM25 index built over {total} chunks\n")


def build_ground_truth(items: list[dict]) -> dict[str, set[str]]:
    """Map question id -> set of relevant chunk ids."""
    all_chunks = get_all_chunks()
    prepared = [
        (chunk["id"], chunk["metadata"]["document_id"], normalize(chunk["text"]))
        for chunk in all_chunks
    ]

    truth: dict[str, set[str]] = {}
    for item in items:
        if not item["answerable"]:
            truth[item["id"]] = set()
            continue
        phrases = [normalize(p) for p in item["evidence_phrases"]]
        truth[item["id"]] = {
            chunk_id
            for chunk_id, document_id, text in prepared
            if document_id == item["document"] and any(p in text for p in phrases)
        }
    return truth


def score(retrieved: list[str], relevant: set[str], k: int) -> dict:
    top = retrieved[:k]
    hits = [chunk_id for chunk_id in top if chunk_id in relevant]
    first_rank = next(
        (i + 1 for i, chunk_id in enumerate(retrieved) if chunk_id in relevant), None
    )
    return {
        "precision": len(hits) / k if k else 0.0,
        "recall": len(hits) / len(relevant) if relevant else 0.0,
        "hit": 1.0 if hits else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    spec = json.loads((EVAL_DIR / "eval_set.json").read_text())
    items = spec["items"]
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]

    index_corpus()
    truth = build_ground_truth(items)

    missing = [i["id"] for i in answerable if not truth[i["id"]]]
    if missing:
        raise SystemExit(
            "These questions matched no chunk -- their evidence phrases are not "
            f"present in the corpus: {missing}"
        )

    print(
        f"Eval set: {len(answerable)} answerable, {len(unanswerable)} unanswerable"
    )
    print(
        "Mean relevant chunks per answerable question: "
        f"{mean([len(truth[i['id']]) for i in answerable]):.2f}\n"
    )

    results: dict = {"configs": {}, "confidence": {}, "per_category": {}}

    # ---------------- Retrieval quality by configuration ----------------
    for config in CONFIGS:
        max_k = max(K_VALUES)
        per_question = []

        for item in answerable:
            result = retrieve(
                item["question"],
                top_k=max_k,
                alpha=config["alpha"],
                use_reranker=config["use_reranker"],
            )
            retrieved = [chunk["id"] for chunk in result.chunks]
            per_question.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "retrieved": retrieved,
                    "scores": {k: score(retrieved, truth[item["id"]], k) for k in K_VALUES},
                }
            )

        summary = {}
        for k in K_VALUES:
            summary[k] = {
                metric: round(
                    mean([q["scores"][k][metric] for q in per_question]), 4
                )
                for metric in ("precision", "recall", "hit", "mrr")
            }

        results["configs"][config["name"]] = {
            "alpha": config["alpha"],
            "use_reranker": config["use_reranker"],
            "summary": summary,
            "per_question": per_question,
        }

        line = "  ".join(
            f"P@{k}={summary[k]['precision']:.3f} R@{k}={summary[k]['recall']:.3f}"
            for k in K_VALUES
        )
        print(f"{config['name']:<18} {line}  MRR={summary[K_VALUES[0]]['mrr']:.3f}")

    # ---------------- Per-category breakdown (full pipeline) -------------
    full = results["configs"]["Hybrid + rerank"]["per_question"]
    vector_only = results["configs"]["Vector only"]["per_question"]
    bm25_only = results["configs"]["BM25 only"]["per_question"]

    for category in ("lexical", "semantic", "mixed"):
        def hit_rate(rows):
            selected = [r for r in rows if r["category"] == category]
            return round(mean([r["scores"][3]["hit"] for r in selected]), 4)

        results["per_category"][category] = {
            "vector_only_hit@3": hit_rate(vector_only),
            "bm25_only_hit@3": hit_rate(bm25_only),
            "hybrid_rerank_hit@3": hit_rate(full),
            "n": len([r for r in full if r["category"] == category]),
        }

    # ---------------- Confidence separation (threshold calibration) ------
    #
    # The threshold is chosen from data, not guessed. Two error types trade off:
    #   * false abstention  -- refusing a question we could have answered
    #   * failure to abstain -- answering from irrelevant context (hallucination
    #     risk), which is the more damaging error for this product
    # so the sweep below reports both at each candidate threshold.
    per_question_conf = []
    for item in items:
        result = retrieve(item["question"], top_k=settings.top_k, use_reranker=True)
        per_question_conf.append(
            {
                "id": item["id"],
                "category": item["category"],
                "answerable": item["answerable"],
                "confidence": round(result.confidence, 4),
                "semantic_confidence": round(result.semantic_confidence, 4),
            }
        )

    answerable_rows = [p for p in per_question_conf if p["answerable"]]
    unanswerable_rows = [p for p in per_question_conf if not p["answerable"]]
    answerable_conf = [p["confidence"] for p in answerable_rows]
    unanswerable_conf = [p["confidence"] for p in unanswerable_rows]

    def abstains(row: dict, t_rerank: float, t_vector: float) -> bool:
        """The shipped gate: abstain only when BOTH signals are weak."""
        return (
            row["confidence"] < t_rerank
            and row["semantic_confidence"] < t_vector
        )

    # Sweep the 2D gate. t_vector = 1.01 degenerates to a rerank-only gate
    # (cosine never exceeds 1), which is the single-signal baseline.
    sweep = []
    for t_rerank in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for t_vector in [0.25, 0.30, 0.35, 0.40, 1.01]:
            false_abstain = sum(
                1 for r in answerable_rows if abstains(r, t_rerank, t_vector)
            )
            correct_abstain = sum(
                1 for r in unanswerable_rows if abstains(r, t_rerank, t_vector)
            )
            sweep.append(
                {
                    "t_rerank": t_rerank,
                    "t_vector": t_vector,
                    "rerank_only": t_vector > 1.0,
                    "correct_abstain": correct_abstain,
                    "correct_abstain_rate": round(
                        correct_abstain / len(unanswerable_rows), 4
                    ),
                    "false_abstain": false_abstain,
                    "false_abstain_rate": round(
                        false_abstain / len(answerable_rows), 4
                    ),
                }
            )

    # Operating point: catch as many unanswerable questions as possible while
    # keeping false refusals at or below 15% -- refusing to answer a question
    # the corpus *does* cover is the error users notice most.
    acceptable = [s for s in sweep if s["false_abstain_rate"] <= 0.15]
    recommended = max(
        acceptable or sweep,
        key=lambda s: (s["correct_abstain"], -s["false_abstain"]),
    )

    def percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return round(ordered[idx], 4)

    answerable_vec = [p["semantic_confidence"] for p in answerable_rows]
    unanswerable_vec = [p["semantic_confidence"] for p in unanswerable_rows]

    results["confidence"] = {
        "answerable": {
            "rerank_mean": round(mean(answerable_conf), 4),
            "rerank_median": percentile(answerable_conf, 0.50),
            "rerank_p10": percentile(answerable_conf, 0.10),
            "vector_mean": round(mean(answerable_vec), 4),
            "vector_min": round(min(answerable_vec), 4),
            "n": len(answerable_rows),
        },
        "unanswerable": {
            "rerank_mean": round(mean(unanswerable_conf), 4),
            "rerank_max": round(max(unanswerable_conf), 4),
            "vector_mean": round(mean(unanswerable_vec), 4),
            "vector_max": round(max(unanswerable_vec), 4),
            "n": len(unanswerable_rows),
        },
        "sweep": sweep,
        "recommended": recommended,
        "configured": {
            "t_rerank": settings.confidence_threshold,
            "t_vector": settings.vector_confidence_threshold,
            "correct_abstain": sum(
                1
                for r in unanswerable_rows
                if abstains(
                    r,
                    settings.confidence_threshold,
                    settings.vector_confidence_threshold,
                )
            ),
            "false_abstain": sum(
                1
                for r in answerable_rows
                if abstains(
                    r,
                    settings.confidence_threshold,
                    settings.vector_confidence_threshold,
                )
            ),
        },
        "per_question": per_question_conf,
    }

    write_markdown(results, len(answerable), len(unanswerable))
    (EVAL_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))


def write_markdown(results: dict, n_answerable: int, n_unanswerable: int) -> None:
    lines = [
        f"Evaluated on {n_answerable} answerable questions "
        f"({n_unanswerable} additional unanswerable questions used for "
        "threshold calibration) over a 3-document corpus.",
        "",
        "| Configuration | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | hit@3 | MRR |",
        "|---------------|----:|----:|----:|----:|----:|----:|------:|----:|",
    ]
    for name, data in results["configs"].items():
        s = data["summary"]
        lines.append(
            f"| {name} | {s[1]['precision']:.3f} | {s[1]['recall']:.3f} | "
            f"{s[3]['precision']:.3f} | {s[3]['recall']:.3f} | "
            f"{s[5]['precision']:.3f} | {s[5]['recall']:.3f} | "
            f"{s[3]['hit']:.3f} | {s[3]['mrr']:.3f} |"
        )

    lines += [
        "",
        "**hit@3 by question type** — where each retriever earns its place:",
        "",
        "| Question type | n | Vector only | BM25 only | Hybrid + rerank |",
        "|---------------|--:|------------:|----------:|----------------:|",
    ]
    for category, data in results["per_category"].items():
        lines.append(
            f"| {category} | {data['n']} | {data['vector_only_hit@3']:.3f} | "
            f"{data['bm25_only_hit@3']:.3f} | {data['hybrid_rerank_hit@3']:.3f} |"
        )

    conf = results["confidence"]
    a, u, rec, cfg = (
        conf["answerable"],
        conf["unanswerable"],
        conf["recommended"],
        conf["configured"],
    )
    lines += [
        "",
        "### Abstention calibration",
        "",
        "Neither confidence signal separates answerable from unanswerable on "
        "its own:",
        "",
        "| | n | cross-encoder mean | median | p10 | cosine mean | cosine max |",
        "|---|--:|-------------------:|-------:|----:|------------:|-----------:|",
        f"| Answerable | {a['n']} | {a['rerank_mean']:.3f} | "
        f"{a['rerank_median']:.3f} | {a['rerank_p10']:.3f} | "
        f"{a['vector_mean']:.3f} | — |",
        f"| Unanswerable | {u['n']} | {u['rerank_mean']:.3f} | — | — | "
        f"{u['vector_mean']:.3f} | {u['vector_max']:.3f} |",
        "",
        "The cross-encoder has a **median of "
        f"{a['rerank_median']:.3f} but a p10 of {a['rerank_p10']:.3f}** on "
        "answerable questions — bimodal, because it scores correct-but-"
        "paraphrased passages near zero. Cosine survives paraphrase but rates "
        "topically-adjacent irrelevant chunks highly (max "
        f"{u['vector_max']:.3f} on unanswerable, above many answerable ones).",
        "",
        "Because they fail on *different* questions, the shipped gate abstains "
        "only when **both** are weak. `rerank-only` rows below are the "
        "single-signal baseline:",
        "",
        "| t_rerank | t_cosine | Catches unanswerable | Wrongly refuses answerable |",
        "|---------:|---------:|---------------------:|---------------------------:|",
    ]
    for row in conf["sweep"]:
        # Show the recommended point, its rerank-only counterpart, and the
        # single-signal baselines -- the full 30-row grid is in the JSON.
        is_rec = (
            row["t_rerank"] == rec["t_rerank"]
            and row["t_vector"] == rec["t_vector"]
        )
        if not (is_rec or row["rerank_only"]):
            continue
        label = "rerank only" if row["rerank_only"] else f"{row['t_vector']:.2f}"
        mark = " ✅" if is_rec else ""
        lines.append(
            f"| {row['t_rerank']:.2f} | {label}{mark} | "
            f"{row['correct_abstain']}/{u['n']} "
            f"({row['correct_abstain_rate']:.0%}) | "
            f"{row['false_abstain']}/{a['n']} "
            f"({row['false_abstain_rate']:.0%}) |"
        )

    lines += [
        "",
        f"**Shipped defaults: `CONFIDENCE_THRESHOLD={cfg['t_rerank']}`, "
        f"`VECTOR_CONFIDENCE_THRESHOLD={cfg['t_vector']}`** — catches "
        f"{cfg['correct_abstain']}/{u['n']} unanswerable questions while "
        f"wrongly refusing {cfg['false_abstain']}/{a['n']} answerable ones.",
        "",
        "The gate is a cheap pre-filter, not the only guardrail: questions that "
        "pass it still reach a model instructed to reply "
        '"I don\'t have enough information" when the excerpts don\'t cover the '
        "question. The threshold exists to avoid paying for that call when "
        "retrieval clearly found nothing.",
        "",
        f"> Calibrated on only {u['n']} unanswerable questions — enough to show "
        "the single-signal gate is unusable, not enough to fix these "
        "thresholds precisely. Re-run against your own corpus before relying "
        "on them.",
    ]

    text = "\n".join(lines)
    (EVAL_DIR / "eval_results.md").write_text(text + "\n")
    print(f"\nResults -> {EVAL_DIR / 'eval_results.md'}\n")
    print(text)


if __name__ == "__main__":
    main()
