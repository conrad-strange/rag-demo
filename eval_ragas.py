import argparse
import copy
import json
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    FINAL_TOP_K,
    LOG_DIR,
    LLM_TEMPERATURE,
    USE_HYBRID_SEARCH,
    USE_RERANK,
    VECTOR_TOP_K,
    ensure_dirs,
)
from agent_router import AgentRAGWorkflow
from rag_pipline import RAGPipeline


DEFAULT_EVAL_PATH = os.path.join("data", "retrieval_eval_questions.csv")
DEFAULT_DATASET_PATH = os.path.join(LOG_DIR, "ragas_eval_dataset.jsonl")
DEFAULT_RESULT_PATH = os.path.join(LOG_DIR, "ragas_eval_results.csv")
DEFAULT_SUMMARY_PATH = os.path.join(LOG_DIR, "ragas_eval_summary.json")

DEFAULT_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall,answer_correctness"
REFERENCE_COLUMNS = ("reference", "reference_answer", "ground_truth", "expected_answer")
REFERENCE_REQUIRED_METRICS = {"context_precision", "context_recall", "answer_correctness"}


@contextmanager
def silence_output(enabled: bool):
    if not enabled:
        with nullcontext():
            yield
        return

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def quiet_model_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def patch_ragas_mistral_import() -> None:
    """
    RAGAS 0.4.3 imports instructor, and instructor 1.15.x expects
    `from mistralai import Mistral`. mistralai 2.x keeps Mistral under
    `mistralai.client.sdk`. Expose the old top-level name at runtime.
    """
    try:
        import mistralai

        if not hasattr(mistralai, "Mistral"):
            from mistralai.client.sdk import Mistral

            mistralai.Mistral = Mistral
    except Exception:
        # Let the real RAGAS import surface the actionable error.
        return


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


def first_available_reference(row: pd.Series) -> Optional[str]:
    for column in REFERENCE_COLUMNS:
        if column in row.index and not pd.isna(row[column]):
            text = str(row[column]).strip()
            if text:
                return text
    return None


def synthetic_reference(row: pd.Series) -> str:
    sources = split_cell(row.get("expected_sources", ""))
    partial_sources = split_cell(row.get("partial_sources", ""))
    keywords = split_cell(row.get("expected_keywords", ""))
    notes = str(row.get("notes", "") or "").strip()

    parts = []
    if sources:
        parts.append("The answer should be grounded mainly in: " + ", ".join(sources) + ".")
    if partial_sources:
        parts.append("Partially relevant supporting sources include: " + ", ".join(partial_sources) + ".")
    if keywords:
        parts.append("The answer should cover these concepts: " + ", ".join(keywords) + ".")
    if notes:
        parts.append("Evaluation note: " + notes + ".")
    return " ".join(parts) or "The answer should be grounded in the retrieved research-paper context."


def load_eval_rows(eval_path: str, include_negative: bool, limit: Optional[int]) -> pd.DataFrame:
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_path}")

    df = pd.read_csv(eval_path, encoding="utf-8-sig")
    required = {"query_id", "question", "is_answerable"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV missing columns: {sorted(missing)}")

    if not include_negative:
        df = df[df["is_answerable"].apply(lambda value: as_bool(value, default=True))]

    if limit is not None:
        df = df.head(limit)

    return df.reset_index(drop=True)


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model=None, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = str(model_name)
        self._model = model

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def ragas_imports():
    patch_ragas_mistral_import()
    warnings.filterwarnings(
        "ignore",
        message=r"Importing .* from 'ragas.metrics' is deprecated.*",
        category=DeprecationWarning,
    )

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics._answer_correctness import answer_correctness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    return evaluate, EvaluationDataset, metric_map


def parse_metrics(metric_text: str) -> List[str]:
    metrics = [item.strip() for item in metric_text.split(",") if item.strip()]
    if not metrics:
        raise ValueError("At least one RAGAS metric is required.")
    return metrics


def validate_metric_config(
    metric_names: Sequence[str],
    df: pd.DataFrame,
    use_synthetic_reference: bool,
) -> None:
    _, _, metric_map = ragas_imports()
    unknown = sorted(set(metric_names) - set(metric_map))
    if unknown:
        raise ValueError(f"Unknown RAGAS metrics: {unknown}. Available: {sorted(metric_map)}")

    needs_reference = sorted(set(metric_names) & REFERENCE_REQUIRED_METRICS)
    if not needs_reference:
        return

    reference_count = sum(1 for _, row in df.iterrows() if first_available_reference(row))
    if use_synthetic_reference:
        return
    if reference_count != len(df):
        raise ValueError(
            "Metrics "
            + ", ".join(needs_reference)
            + " require a reference answer for every evaluated row. "
            + f"Found references for {reference_count}/{len(df)} rows. "
            + "Add a reference_answer column or pass --use-synthetic-reference for a rough experiment."
        )


def compact_sources(docs: Sequence[Dict]) -> str:
    items = []
    for rank, doc in enumerate(docs, start=1):
        items.append(
            f"{rank}:{doc.get('source')}#chunk{doc.get('chunk_id')}"
            f"#vector{doc.get('vector_id')}"
            f"(score={doc.get('vector_score', 0.0):.4f})"
        )
    return ";".join(items)


def agent_task_from_query_type(query_type: str) -> Optional[str]:
    normalized = str(query_type or "").strip().lower()
    if normalized == "compare":
        return "compare"
    if normalized in {"synthesis", "literature_review"}:
        return "summary"
    return None


def build_ragas_dataset_rows(
    df: pd.DataFrame,
    top_k: int,
    candidate_k: int,
    use_hybrid: bool,
    use_rerank: bool,
    use_agent: bool,
    use_synthetic_reference: bool,
    save_query_log: bool,
    verbose: bool,
) -> Tuple[List[Dict], List[Dict], Optional[object]]:
    with silence_output(not verbose):
        rag = RAGPipeline(use_rerank=use_rerank, load_llm_client=True)
        agent = AgentRAGWorkflow(rag) if use_agent else None

    samples = []
    metadata = []
    candidate_k = max(candidate_k, top_k)

    for index, row in df.iterrows():
        query_id = str(row["query_id"])
        question = str(row["question"])
        category = str(row.get("category", "all") or "all").strip() or "all"
        query_type = row.get("query_type", "")
        forced_task_type = agent_task_from_query_type(query_type) if use_agent else None

        if verbose:
            print(f"[{index + 1}/{len(df)}] {query_id}: {question}")

        with silence_output(not verbose):
            if agent is not None:
                result = agent.agent_answer(
                    query=question,
                    category=category,
                    vector_top_k=candidate_k,
                    final_top_k=top_k,
                    use_hybrid=use_hybrid,
                    save_log=save_query_log,
                    forced_task_type=forced_task_type,
                )
            else:
                result = rag.answer(
                    query=question,
                    category=category,
                    vector_top_k=candidate_k,
                    final_top_k=top_k,
                    use_hybrid=use_hybrid,
                    save_log=save_query_log,
                )

        retrieved_docs = result.get("retrieved_docs", [])
        sample = {
            "user_input": question,
            "response": result.get("answer", ""),
            "retrieved_contexts": [str(doc.get("text", "")) for doc in retrieved_docs],
        }

        reference = first_available_reference(row)
        if reference is None and use_synthetic_reference:
            reference = synthetic_reference(row)
        if reference:
            sample["reference"] = reference

        samples.append(sample)
        metadata.append({
            "query_id": query_id,
            "question": question,
            "query_type": query_type,
            "difficulty": row.get("difficulty", ""),
            "category": category,
            "answer_mode": "agent" if use_agent else "normal",
            "forced_task_type": forced_task_type or "",
            "task_type": result.get("task_type", ""),
            "tool_used": result.get("tool_used", ""),
            "status": result.get("status", ""),
            "best_score": result.get("best_score", None),
            "configured_candidate_k": candidate_k,
            "configured_top_k": top_k,
            "retrieved_count": len(retrieved_docs),
            "expected_sources": row.get("expected_sources", ""),
            "partial_sources": row.get("partial_sources", ""),
            "expected_keywords": row.get("expected_keywords", ""),
            "retrieved_sources": compact_sources(retrieved_docs),
        })

    return samples, metadata, rag.embedder


def save_dataset_jsonl(samples: Sequence[Dict], metadata: Sequence[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample, meta in zip(samples, metadata):
            f.write(json.dumps({"sample": sample, "metadata": meta}, ensure_ascii=False) + "\n")


def load_dataset_jsonl(path: str) -> Tuple[List[Dict], List[Dict]]:
    samples = []
    metadata = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append(item["sample"])
            metadata.append(item.get("metadata", {}))
    return samples, metadata


def make_evaluator_llm(model: str, base_url: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for RAGAS evaluator LLM calls.")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def summarize_scores(df: pd.DataFrame, metric_names: Sequence[str]) -> Dict:
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": int(len(df)),
        "metrics": list(metric_names),
    }
    for metric in metric_names:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            summary[metric] = float(values.mean()) if len(values) else None
    return summary


def print_summary(summary: Dict, result_path: str, summary_path: str, dataset_path: str) -> None:
    print("\n========== RAGAS Answer Evaluation Result ==========")
    print(f"mode: {'agent' if summary.get('use_agent') else 'normal'}")
    print(f"rows: {summary['row_count']}")
    print("metrics: " + ", ".join(summary["metrics"]))
    for metric in summary["metrics"]:
        value = summary.get(metric)
        print(f"{metric}: {'n/a' if value is None else f'{value:.3f}'}")
    print(f"dataset_jsonl: {dataset_path}")
    print(f"results_csv: {result_path}")
    print(f"summary_json: {summary_path}")


def evaluate_ragas(
    eval_path: str = DEFAULT_EVAL_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    result_path: str = DEFAULT_RESULT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    metric_text: str = DEFAULT_METRICS,
    top_k: int = FINAL_TOP_K,
    candidate_k: int = VECTOR_TOP_K,
    use_hybrid: bool = USE_HYBRID_SEARCH,
    use_rerank: bool = USE_RERANK,
    use_agent: bool = False,
    include_negative: bool = False,
    use_synthetic_reference: bool = False,
    reuse_dataset: Optional[str] = None,
    limit: Optional[int] = None,
    evaluator_model: str = DEEPSEEK_MODEL_NAME,
    evaluator_base_url: str = DEEPSEEK_BASE_URL,
    evaluator_temperature: float = LLM_TEMPERATURE,
    batch_size: Optional[int] = None,
    save_query_log: bool = False,
    verbose: bool = False,
    validate_only: bool = False,
) -> Dict:
    ensure_dirs()
    load_dotenv(override=True)
    if not verbose:
        quiet_model_libraries()

    metric_names = parse_metrics(metric_text)
    eval_df = load_eval_rows(eval_path, include_negative=include_negative, limit=limit)
    validate_metric_config(metric_names, eval_df, use_synthetic_reference=use_synthetic_reference)
    evaluate, EvaluationDataset, metric_map = ragas_imports()

    if validate_only:
        summary = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "validated",
            "eval_path": eval_path,
            "row_count": int(len(eval_df)),
            "metrics": metric_names,
            "ragas_import": "ok",
            "mistralai_compat_patch": "active_if_needed",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    embedder = None
    if reuse_dataset:
        samples, metadata = load_dataset_jsonl(reuse_dataset)
        dataset_path = reuse_dataset
    else:
        samples, metadata, embedder = build_ragas_dataset_rows(
            df=eval_df,
            top_k=top_k,
            candidate_k=candidate_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_agent=use_agent,
            use_synthetic_reference=use_synthetic_reference,
            save_query_log=save_query_log,
            verbose=verbose,
        )
        save_dataset_jsonl(samples, metadata, dataset_path)

    metrics = []
    for name in metric_names:
        metric = copy.deepcopy(metric_map[name])
        if name == "answer_relevancy" and getattr(metric, "strictness", None) != 1:
            # DeepSeek's OpenAI-compatible endpoint currently supports n=1 only.
            # RAGAS defaults answer_relevancy.strictness to 3, which requests n=3.
            metric.strictness = 1
        metrics.append(metric)
    evaluator_llm = make_evaluator_llm(
        model=evaluator_model,
        base_url=evaluator_base_url,
        temperature=evaluator_temperature,
    )
    evaluator_embeddings = None
    if any(name in {"answer_relevancy", "answer_correctness"} for name in metric_names):
        evaluator_embeddings = SentenceTransformerEmbeddings(model=embedder)

    dataset = EvaluationDataset.from_list(samples)
    with silence_output(not verbose):
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            show_progress=verbose,
            batch_size=batch_size,
            raise_exceptions=False,
        )

    result_df = result.to_pandas()
    meta_df = pd.DataFrame(metadata)
    result_df = pd.concat([meta_df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = summarize_scores(result_df, metric_names)
    summary.update({
        "eval_path": eval_path,
        "dataset_path": dataset_path,
        "result_path": result_path,
        "summary_path": summary_path,
        "top_k": top_k,
        "candidate_k": max(candidate_k, top_k),
        "use_hybrid": use_hybrid,
        "use_rerank": use_rerank,
        "use_agent": use_agent,
        "agent_query_type_routing": use_agent,
        "include_negative": include_negative,
        "use_synthetic_reference": use_synthetic_reference,
        "evaluator_model": evaluator_model,
        "evaluator_base_url": evaluator_base_url,
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary, result_path, summary_path, dataset_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer-level RAGAS evaluation over the paper RAG system.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_PATH, help="CSV file with evaluation questions.")
    parser.add_argument("--dataset-out", default=DEFAULT_DATASET_PATH, help="Generated RAGAS dataset JSONL path.")
    parser.add_argument("--reuse-dataset", default=None, help="Reuse a previously generated RAGAS dataset JSONL.")
    parser.add_argument("--out", default=DEFAULT_RESULT_PATH, help="Output CSV path with per-question RAGAS scores.")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_PATH, help="Output JSON path with summary scores.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Comma-separated RAGAS metrics.")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K, help="Final context count used for RAG answers.")
    parser.add_argument("--candidate-k", type=int, default=VECTOR_TOP_K, help="Initial retrieval candidate count.")
    parser.add_argument("--hybrid", action="store_true", help="Use BM25 + vector hybrid retrieval for answer generation.")
    parser.add_argument("--agent", action="store_true", help="Use AgentRAGWorkflow for answer generation.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable rerank during answer generation.")
    parser.add_argument("--include-negative", action="store_true", help="Also evaluate rows marked is_answerable=no.")
    parser.add_argument(
        "--use-synthetic-reference",
        action="store_true",
        help="Build weak references from expected sources/keywords for reference-based RAGAS metrics.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N selected rows.")
    parser.add_argument("--evaluator-model", default=DEEPSEEK_MODEL_NAME, help="LLM model used by RAGAS judge.")
    parser.add_argument("--evaluator-base-url", default=DEEPSEEK_BASE_URL, help="OpenAI-compatible evaluator base URL.")
    parser.add_argument("--evaluator-temperature", type=float, default=LLM_TEMPERATURE, help="Evaluator LLM temperature.")
    parser.add_argument("--batch-size", type=int, default=None, help="RAGAS evaluation batch size.")
    parser.add_argument("--save-query-log", action="store_true", help="Write normal RAG query logs while generating answers.")
    parser.add_argument("--verbose", action="store_true", help="Print answer generation and RAGAS progress.")
    parser.add_argument("--validate-only", action="store_true", help="Validate CSV, imports, and metric config without API calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        evaluate_ragas(
            eval_path=args.eval_file,
            dataset_path=args.dataset_out,
            result_path=args.out,
            summary_path=args.summary_out,
            metric_text=args.metrics,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            use_hybrid=args.hybrid,
            use_rerank=not args.no_rerank,
            use_agent=args.agent,
            include_negative=args.include_negative,
            use_synthetic_reference=args.use_synthetic_reference,
            reuse_dataset=args.reuse_dataset,
            limit=args.limit,
            evaluator_model=args.evaluator_model,
            evaluator_base_url=args.evaluator_base_url,
            evaluator_temperature=args.evaluator_temperature,
            batch_size=args.batch_size,
            save_query_log=args.save_query_log,
            verbose=args.verbose,
            validate_only=args.validate_only,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
