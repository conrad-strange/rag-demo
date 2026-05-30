import argparse
import json
import math
import os
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from config import FINAL_TOP_K, LOG_DIR, USE_HYBRID_SEARCH, VECTOR_TOP_K, ensure_dirs
from rag_pipline import RAGPipeline


DEFAULT_EVAL_PATH = os.path.join("data", "retrieval_eval_questions.csv")
DEFAULT_RESULT_PATH = os.path.join(LOG_DIR, "retrieval_eval_results.csv")
DEFAULT_SUMMARY_PATH = os.path.join(LOG_DIR, "retrieval_eval_summary.json")


@contextmanager
def silence_output(enabled: bool):
    if not enabled:
        with nullcontext():
            yield
        return

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def split_cell(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.replace("|", ";")
    return [item.strip() for item in text.split(";") if item.strip()]


def as_bool(value, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    return str(value).strip().lower() not in {"no", "false", "0", "n"}


def normalize_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def doc_keys(doc: Dict) -> Set[str]:
    keys = set()
    for field in ["source", "relative_path", "path", "doc_id"]:
        value = doc.get(field)
        if value:
            normalized = normalize_key(str(value))
            keys.add(normalized)
            keys.add(Path(normalized).name)
    return keys


def source_matches(doc: Dict, sources: Sequence[str]) -> Set[str]:
    expected = {normalize_key(item) for item in sources}
    return doc_keys(doc) & expected


def relevance_grade(doc: Dict, primary_sources: Sequence[str], partial_sources: Sequence[str]) -> float:
    if source_matches(doc, primary_sources):
        return 1.0
    if source_matches(doc, partial_sources):
        return 0.5
    return 0.0


def grades_for_docs(
    docs: Sequence[Dict],
    primary_sources: Sequence[str],
    partial_sources: Sequence[str],
) -> List[float]:
    return [relevance_grade(doc, primary_sources, partial_sources) for doc in docs]


def matched_sources(docs: Sequence[Dict], sources: Sequence[str]) -> Set[str]:
    matched = set()
    for doc in docs:
        matched.update(source_matches(doc, sources))
    return matched


def source_novelty_grades(
    docs: Sequence[Dict],
    primary_sources: Sequence[str],
    partial_sources: Sequence[str],
) -> List[float]:
    seen_primary = set()
    seen_partial = set()
    grades = []
    for doc in docs:
        primary = source_matches(doc, primary_sources) - seen_primary
        partial = source_matches(doc, partial_sources) - seen_partial
        if primary:
            grades.append(1.0)
        elif partial:
            grades.append(0.5)
        else:
            grades.append(0.0)
        seen_primary.update(primary)
        seen_partial.update(partial)
    return grades


def reciprocal_rank(grades: Sequence[float], minimum_grade: float = 1.0) -> float:
    for rank, grade in enumerate(grades, start=1):
        if grade >= minimum_grade:
            return 1.0 / rank
    return 0.0


def dcg(grades: Sequence[float]) -> float:
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(grades: Sequence[float], ideal_grades: Sequence[float], k: int) -> Optional[float]:
    if not ideal_grades:
        return None
    ideal = sorted(ideal_grades, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(grades[:k]) / ideal_dcg if ideal_dcg else 0.0


def keyword_hit_rate(retrieved_docs: Sequence[Dict], expected_keywords: Sequence[str]) -> Optional[float]:
    if not expected_keywords:
        return None
    text = "\n".join(str(doc.get("text", "")) for doc in retrieved_docs).lower()
    hits = [keyword for keyword in expected_keywords if keyword.lower() in text]
    return len(hits) / len(expected_keywords)


def compact_sources(docs: Sequence[Dict]) -> str:
    items = []
    for rank, doc in enumerate(docs, start=1):
        items.append(
            f"{rank}:{doc.get('source')}#chunk{doc.get('chunk_id')}"
            f"#vector{doc.get('vector_id')}"
            f"(score={doc.get('vector_score', 0.0):.4f})"
        )
    return ";".join(items)


def retrieve_candidates(
    rag: RAGPipeline,
    query: str,
    category: str,
    candidate_k: int,
    use_hybrid: bool,
) -> List[Dict]:
    if use_hybrid:
        return rag.hybrid_retrieve(
            query=query,
            vector_top_k=candidate_k,
            category=category,
        )
    return rag.vector_retrieve(
        query=query,
        top_k=candidate_k,
        category=category,
    )


def final_docs_from_candidates(
    rag: RAGPipeline,
    query: str,
    candidates: Sequence[Dict],
    top_k: int,
    use_rerank: bool,
) -> List[Dict]:
    if use_rerank:
        return rag.rerank(query, list(candidates), final_top_k=top_k)
    return list(candidates[:top_k])


def score_margin(docs: Sequence[Dict], grades: Sequence[float]) -> Optional[float]:
    relevant_scores = [
        float(doc.get("vector_score", 0.0))
        for doc, grade in zip(docs, grades)
        if grade > 0
    ]
    noise_scores = [
        float(doc.get("vector_score", 0.0))
        for doc, grade in zip(docs, grades)
        if grade == 0
    ]
    if not relevant_scores or not noise_scores:
        return None
    return max(relevant_scores) - max(noise_scores)


def mean_or_none(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    return float(values.mean()) if len(values) else None


def format_metric(value) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def group_summary(df: pd.DataFrame, group_field: str, top_k: int, candidate_k: int) -> Dict:
    if group_field not in df.columns or df.empty:
        return {}
    rows = {}
    for name, group in df.groupby(group_field):
        rows[str(name)] = {
            "count": int(len(group)),
            f"hit_rate_at_{top_k}": mean_or_none(group[f"hit_at_{top_k}"]),
            "top1_primary_accuracy": mean_or_none(group["top1_primary_relevant"]),
            f"avg_primary_source_recall_at_{top_k}": mean_or_none(group[f"primary_source_recall_at_{top_k}"]),
            f"avg_candidate_noise_rate_at_{candidate_k}": mean_or_none(group[f"candidate_noise_rate_at_{candidate_k}"]),
            "mrr_primary": mean_or_none(group["mrr_primary"]),
        }
    return rows


def weakest_groups(groups: Dict, metric_key: str, limit: int = 3) -> List[str]:
    ranked = []
    for name, values in groups.items():
        metric = values.get(metric_key)
        if metric is None:
            continue
        ranked.append((float(metric), name, values))
    ranked.sort(key=lambda item: item[0])
    lines = []
    for metric, name, values in ranked[:limit]:
        lines.append(
            f"{name} "
            f"(count={values.get('count')}, {metric_key}={format_metric(metric)})"
        )
    return lines


def print_concise_summary(summary: Dict) -> None:
    top_k = summary["top_k"]
    candidate_k = summary["candidate_k"]
    hit_key = f"hit_rate_at_{top_k}"
    candidate_hit_key = f"candidate_hit_rate_at_{candidate_k}"
    recall_key = f"avg_primary_source_recall_at_{top_k}"
    graded_recall_key = f"avg_graded_recall_at_{top_k}"
    precision_key = f"avg_graded_chunk_precision_at_{top_k}"
    final_noise_key = f"avg_final_noise_rate_at_{top_k}"
    candidate_noise_key = f"avg_candidate_noise_rate_at_{candidate_k}"
    ndcg_key = f"avg_ndcg_at_{top_k}"

    print("\n========== Retrieval Evaluation Result ==========")
    print(
        "Config: "
        f"top_k={top_k}, candidate_k={candidate_k}, "
        f"hybrid={summary['use_hybrid']}, rerank={summary['use_rerank']}"
    )
    print(
        "Questions: "
        f"{summary['question_count']} total, "
        f"{summary['positive_question_count']} answerable, "
        f"{summary['negative_question_count']} negative"
    )
    print(f"{hit_key}: {format_metric(summary.get(hit_key))}")
    print(f"{candidate_hit_key}: {format_metric(summary.get(candidate_hit_key))}")
    print(f"top1_primary_accuracy: {format_metric(summary.get('top1_primary_accuracy'))}")
    print(f"top1_graded_accuracy: {format_metric(summary.get('top1_graded_accuracy'))}")
    print(f"{recall_key}: {format_metric(summary.get(recall_key))}")
    print(f"{graded_recall_key}: {format_metric(summary.get(graded_recall_key))}")
    print(f"{precision_key}: {format_metric(summary.get(precision_key))}")
    print(f"{final_noise_key}: {format_metric(summary.get(final_noise_key))}")
    print(f"{candidate_noise_key}: {format_metric(summary.get(candidate_noise_key))}")
    print(f"mrr_primary: {format_metric(summary.get('mrr_primary'))}")
    print(f"{ndcg_key}: {format_metric(summary.get(ndcg_key))}")
    print(f"negative_high_confidence_rate: {format_metric(summary.get('negative_high_confidence_rate'))}")

    weak_by_type = weakest_groups(summary.get("by_query_type", {}), recall_key)
    weak_by_difficulty = weakest_groups(summary.get("by_difficulty", {}), recall_key)
    if weak_by_type:
        print("weakest_query_types_by_recall: " + " | ".join(weak_by_type))
    if weak_by_difficulty:
        print("weakest_difficulties_by_recall: " + " | ".join(weak_by_difficulty))

    print(f"results_csv: {summary['result_path']}")
    print(f"summary_json: {summary['summary_path']}")


def evaluate_retrieval(
    eval_path: str = DEFAULT_EVAL_PATH,
    result_path: str = DEFAULT_RESULT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    top_k: int = FINAL_TOP_K,
    candidate_k: int = VECTOR_TOP_K,
    use_hybrid: bool = USE_HYBRID_SEARCH,
    use_rerank: bool = False,
    negative_score_threshold: float = 0.55,
    verbose: bool = False,
) -> Dict:
    ensure_dirs()

    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_path}")

    eval_df = pd.read_csv(eval_path)
    required = {"query_id", "question", "expected_sources"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV missing columns: {sorted(missing)}")

    with silence_output(not verbose):
        rag = RAGPipeline(use_rerank=use_rerank, load_llm_client=False)
    rows = []
    candidate_k = max(candidate_k, top_k)

    for index, row in eval_df.iterrows():
        query_id = row["query_id"]
        question = row["question"]
        category = row.get("category", "all")
        if pd.isna(category) or not str(category).strip():
            category = "all"
        category = str(category).strip()

        query_type = row.get("query_type", "unspecified")
        difficulty = row.get("difficulty", "unspecified")
        is_answerable = as_bool(row.get("is_answerable", "yes"), default=True)
        primary_sources = split_cell(row.get("expected_sources", ""))
        partial_sources = split_cell(row.get("partial_sources", ""))
        expected_keywords = split_cell(row.get("expected_keywords", ""))

        if verbose:
            print(f"[{index + 1}/{len(eval_df)}] {query_id}: {question}")
        with silence_output(not verbose):
            candidates = retrieve_candidates(
                rag=rag,
                query=question,
                category=category,
                candidate_k=candidate_k,
                use_hybrid=use_hybrid,
            )
            retrieved_docs = final_docs_from_candidates(
                rag=rag,
                query=question,
                candidates=candidates,
                top_k=top_k,
                use_rerank=use_rerank,
            )

        final_grades = grades_for_docs(retrieved_docs, primary_sources, partial_sources)
        candidate_grades = grades_for_docs(candidates, primary_sources, partial_sources)
        final_primary = [1 if grade == 1.0 else 0 for grade in final_grades]
        candidate_primary = [1 if grade == 1.0 else 0 for grade in candidate_grades]
        final_source_grades = source_novelty_grades(retrieved_docs, primary_sources, partial_sources)
        candidate_source_grades = source_novelty_grades(candidates, primary_sources, partial_sources)
        ideal_grades = [1.0] * len(set(map(normalize_key, primary_sources))) + [0.5] * len(set(map(normalize_key, partial_sources)))

        primary_expected_count = len(set(map(normalize_key, primary_sources)))
        partial_expected_count = len(set(map(normalize_key, partial_sources)))
        final_matched_primary = matched_sources(retrieved_docs, primary_sources)
        candidate_matched_primary = matched_sources(candidates, primary_sources)
        final_matched_partial = matched_sources(retrieved_docs, partial_sources)

        retrieved_count = len(retrieved_docs)
        candidate_count = len(candidates)
        best_score = max([float(doc.get("vector_score", 0.0)) for doc in retrieved_docs], default=0.0)
        candidate_best_score = max([float(doc.get("vector_score", 0.0)) for doc in candidates], default=0.0)

        if is_answerable and primary_expected_count > 0:
            hit_at_k = bool(final_matched_primary)
            candidate_hit = bool(candidate_matched_primary)
            primary_source_recall = len(final_matched_primary) / primary_expected_count
            candidate_primary_source_recall = len(candidate_matched_primary) / primary_expected_count
            top1_primary_relevant = bool(final_primary[0]) if final_primary else False
            top1_graded_relevant = bool(final_grades[0] > 0) if final_grades else False
            primary_chunk_precision = sum(final_primary) / retrieved_count if retrieved_count else None
            graded_chunk_precision = sum(final_grades) / retrieved_count if retrieved_count else None
            final_noise_rate = sum(1 for grade in final_grades if grade == 0) / retrieved_count if retrieved_count else None
            candidate_noise_rate = sum(1 for grade in candidate_grades if grade == 0) / candidate_count if candidate_count else None
            graded_recall = sum(final_source_grades) / sum(ideal_grades) if ideal_grades else None
            candidate_graded_recall = sum(candidate_source_grades) / sum(ideal_grades) if ideal_grades else None
            mrr_primary = reciprocal_rank(final_primary, minimum_grade=1.0)
            mrr_graded = reciprocal_rank(final_grades, minimum_grade=0.5)
            ndcg = ndcg_at_k(final_source_grades, ideal_grades, top_k)
            negative_high_confidence = None
            negative_safe = None
        else:
            hit_at_k = None
            candidate_hit = None
            primary_source_recall = None
            candidate_primary_source_recall = None
            top1_primary_relevant = None
            top1_graded_relevant = None
            primary_chunk_precision = None
            graded_chunk_precision = None
            final_noise_rate = 1.0 if retrieved_count else None
            candidate_noise_rate = 1.0 if candidate_count else None
            graded_recall = None
            candidate_graded_recall = None
            mrr_primary = None
            mrr_graded = None
            ndcg = None
            negative_high_confidence = best_score >= negative_score_threshold
            negative_safe = not negative_high_confidence

        rows.append({
            "query_id": query_id,
            "question": question,
            "query_type": query_type,
            "difficulty": difficulty,
            "is_answerable": is_answerable,
            "category": category,
            "expected_sources": ";".join(primary_sources),
            "partial_sources": ";".join(partial_sources),
            "matched_primary_sources": ";".join(sorted(final_matched_primary)),
            "matched_partial_sources": ";".join(sorted(final_matched_partial)),
            "expected_primary_source_count": primary_expected_count,
            "expected_partial_source_count": partial_expected_count,
            "retrieved_count": retrieved_count,
            "candidate_count": candidate_count,
            "best_score": best_score,
            "candidate_best_score": candidate_best_score,
            "negative_high_confidence": negative_high_confidence,
            "negative_safe": negative_safe,
            f"hit_at_{top_k}": hit_at_k,
            f"candidate_hit_at_{candidate_k}": candidate_hit,
            "top1_primary_relevant": top1_primary_relevant,
            "top1_graded_relevant": top1_graded_relevant,
            f"primary_source_recall_at_{top_k}": primary_source_recall,
            f"candidate_primary_source_recall_at_{candidate_k}": candidate_primary_source_recall,
            f"graded_recall_at_{top_k}": graded_recall,
            f"candidate_graded_recall_at_{candidate_k}": candidate_graded_recall,
            f"primary_chunk_precision_at_{top_k}": primary_chunk_precision,
            f"graded_chunk_precision_at_{top_k}": graded_chunk_precision,
            f"final_noise_rate_at_{top_k}": final_noise_rate,
            f"candidate_noise_rate_at_{candidate_k}": candidate_noise_rate,
            "score_margin_relevant_vs_noise": score_margin(candidates, candidate_grades),
            "mrr_primary": mrr_primary,
            "mrr_graded": mrr_graded,
            f"ndcg_at_{top_k}": ndcg,
            "final_grades": ";".join(str(item) for item in final_grades),
            "candidate_grades": ";".join(str(item) for item in candidate_grades),
            "expected_keywords": ";".join(expected_keywords),
            "retrieved_keyword_hit_rate": keyword_hit_rate(retrieved_docs, expected_keywords),
            "retrieved_sources": compact_sources(retrieved_docs),
            "candidate_sources": compact_sources(candidates),
            "notes": row.get("notes", ""),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    positives = result_df[(result_df["is_answerable"] == True) & (result_df["expected_primary_source_count"] > 0)]
    negatives = result_df[result_df["is_answerable"] == False]

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_path": eval_path,
        "result_path": result_path,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "use_hybrid": use_hybrid,
        "use_rerank": use_rerank,
        "negative_score_threshold": negative_score_threshold,
        "answer_level_eval": "not_implemented_retrieval_only",
        "question_count": int(len(result_df)),
        "positive_question_count": int(len(positives)),
        "negative_question_count": int(len(negatives)),
        "avg_best_score_positive": mean_or_none(positives["best_score"]) if len(positives) else None,
        f"hit_rate_at_{top_k}": mean_or_none(positives[f"hit_at_{top_k}"]) if len(positives) else None,
        f"candidate_hit_rate_at_{candidate_k}": mean_or_none(positives[f"candidate_hit_at_{candidate_k}"]) if len(positives) else None,
        "top1_primary_accuracy": mean_or_none(positives["top1_primary_relevant"]) if len(positives) else None,
        "top1_graded_accuracy": mean_or_none(positives["top1_graded_relevant"]) if len(positives) else None,
        f"avg_primary_source_recall_at_{top_k}": mean_or_none(positives[f"primary_source_recall_at_{top_k}"]) if len(positives) else None,
        f"avg_candidate_primary_source_recall_at_{candidate_k}": mean_or_none(positives[f"candidate_primary_source_recall_at_{candidate_k}"]) if len(positives) else None,
        f"avg_graded_recall_at_{top_k}": mean_or_none(positives[f"graded_recall_at_{top_k}"]) if len(positives) else None,
        f"avg_candidate_graded_recall_at_{candidate_k}": mean_or_none(positives[f"candidate_graded_recall_at_{candidate_k}"]) if len(positives) else None,
        f"avg_primary_chunk_precision_at_{top_k}": mean_or_none(positives[f"primary_chunk_precision_at_{top_k}"]) if len(positives) else None,
        f"avg_graded_chunk_precision_at_{top_k}": mean_or_none(positives[f"graded_chunk_precision_at_{top_k}"]) if len(positives) else None,
        f"avg_final_noise_rate_at_{top_k}": mean_or_none(positives[f"final_noise_rate_at_{top_k}"]) if len(positives) else None,
        f"avg_candidate_noise_rate_at_{candidate_k}": mean_or_none(positives[f"candidate_noise_rate_at_{candidate_k}"]) if len(positives) else None,
        "avg_score_margin_relevant_vs_noise": mean_or_none(positives["score_margin_relevant_vs_noise"]) if len(positives) else None,
        "mrr_primary": mean_or_none(positives["mrr_primary"]) if len(positives) else None,
        "mrr_graded": mean_or_none(positives["mrr_graded"]) if len(positives) else None,
        f"avg_ndcg_at_{top_k}": mean_or_none(positives[f"ndcg_at_{top_k}"]) if len(positives) else None,
        "avg_retrieved_keyword_hit_rate": mean_or_none(positives["retrieved_keyword_hit_rate"]) if len(positives) else None,
        "avg_best_score_negative": mean_or_none(negatives["best_score"]) if len(negatives) else None,
        "negative_high_confidence_rate": mean_or_none(negatives["negative_high_confidence"]) if len(negatives) else None,
        "negative_safe_rate": mean_or_none(negatives["negative_safe"]) if len(negatives) else None,
        "by_query_type": group_summary(positives, "query_type", top_k, candidate_k),
        "by_difficulty": group_summary(positives, "difficulty", top_k, candidate_k),
    }
    summary["summary_path"] = summary_path

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_concise_summary(summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality without calling the LLM.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_PATH, help="CSV file with retrieval evaluation questions.")
    parser.add_argument("--out", default=DEFAULT_RESULT_PATH, help="Output CSV path for per-question results.")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_PATH, help="Output JSON path for summary metrics.")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K, help="Number of final chunks to evaluate.")
    parser.add_argument("--candidate-k", type=int, default=VECTOR_TOP_K, help="Initial retrieval candidate count.")
    parser.add_argument("--hybrid", action="store_true", help="Use BM25 + vector hybrid retrieval.")
    parser.add_argument("--rerank", action="store_true", help="Apply rerank before computing final top-k metrics.")
    parser.add_argument("--verbose", action="store_true", help="Print each query while evaluating.")
    parser.add_argument(
        "--negative-score-threshold",
        type=float,
        default=0.55,
        help="Negative queries with best_score above this threshold are counted as high-confidence false positives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_retrieval(
        eval_path=args.eval_file,
        result_path=args.out,
        summary_path=args.summary_out,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        use_hybrid=args.hybrid,
        use_rerank=args.rerank,
        negative_score_threshold=args.negative_score_threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
