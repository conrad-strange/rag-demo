import os
import json
import pandas as pd

from config import (
    ensure_dirs,
    EVAL_FILE_PATH,
    EVAL_RESULT_PATH,
    EVAL_SUMMARY_PATH,
    THRESHOLD_COMPARE_PATH,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    SIMILARITY_THRESHOLD
)
from rag_pipline import RAGPipeline


def keyword_hit(answer: str, expected_keywords: str):
    """
    简单关键词命中率。
    使用大小写不敏感匹配。
    """
    if pd.isna(expected_keywords) or str(expected_keywords).strip() == "":
        return {
            "keywords": [],
            "hit_keywords": [],
            "hit_rate": None
        }

    keywords = [
        kw.strip()
        for kw in str(expected_keywords).split(";")
        if kw.strip()
    ]

    answer_lower = answer.lower()

    hit_keywords = [
        kw for kw in keywords
        if kw.lower() in answer_lower
    ]

    hit_rate = len(hit_keywords) / len(keywords) if keywords else None

    return {
        "keywords": keywords,
        "hit_keywords": hit_keywords,
        "hit_rate": hit_rate
    }


def source_hit(retrieved_docs, expected_source):
    """
    Top-K 来源是否命中。
    """
    if pd.isna(expected_source) or expected_source == "none":
        return None

    retrieved_sources = [doc["source"] for doc in retrieved_docs]
    return expected_source in retrieved_sources


def top1_source_hit(retrieved_docs, expected_source):
    """
    Top1 来源是否命中。
    """
    if pd.isna(expected_source) or expected_source == "none":
        return None

    if not retrieved_docs:
        return False

    return retrieved_docs[0]["source"] == expected_source


def evaluate_once(
    rag: RAGPipeline,
    top_k: int = FINAL_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    save_path: str = EVAL_RESULT_PATH,
    category: str = "all"
):
    """
    单次评估。
    category 默认为 all，也可以指定 incident_response / web_security / llm_security。
    """
    if not os.path.exists(EVAL_FILE_PATH):
        raise FileNotFoundError(
            "没有找到 data/eval_questions.csv，请先创建评估问题文件。"
        )

    eval_df = pd.read_csv(EVAL_FILE_PATH)
    results = []

    for i, row in eval_df.iterrows():
        question = row["question"]
        expected_keywords = row.get("expected_keywords", "")
        expected_source = row.get("expected_source", "none")
        should_answer = row.get("should_answer", "yes")

        print(f"[{i + 1}/{len(eval_df)}] {question}")

        result = rag.answer(
            query=question,
            vector_top_k=VECTOR_TOP_K,
            final_top_k=top_k,
            threshold=threshold,
            category=category,
            save_log=True
        )

        kw_result = keyword_hit(result["answer"], expected_keywords)
        src_hit = source_hit(result["retrieved_docs"], expected_source)
        top1_hit = top1_source_hit(result["retrieved_docs"], expected_source)

        if should_answer == "yes":
            refusal_correct = result["status"] == "answered"
        else:
            refusal_correct = result["status"] == "insufficient_context"

        retrieved_sources = ";".join([
            f"{doc['source']}#chunk{doc['chunk_id']}"
            f"(vector={doc.get('vector_score')},rerank={doc.get('rerank_score')})"
            for doc in result["retrieved_docs"]
        ])

        results.append({
            "question": question,
            "should_answer": should_answer,
            "status": result["status"],
            "best_score": result["best_score"],
            "expected_source": expected_source,
            "source_hit": src_hit,
            "top1_source_hit": top1_hit,
            "expected_keywords": expected_keywords,
            "hit_keywords": ";".join(kw_result["hit_keywords"]),
            "keyword_hit_rate": kw_result["hit_rate"],
            "refusal_correct": refusal_correct,
            "used_rerank": result["used_rerank"],
            "retrieved_sources": retrieved_sources,
            "answer": result["answer"]
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result_df


def summarize(result_df: pd.DataFrame, save_path: str = EVAL_SUMMARY_PATH):
    """
    汇总评估结果。
    """
    answerable_df = result_df[result_df["should_answer"] == "yes"]
    unanswerable_df = result_df[result_df["should_answer"] == "no"]

    summary = {
        "total_questions": len(result_df),
        "answerable_questions": len(answerable_df),
        "unanswerable_questions": len(unanswerable_df),
        "avg_best_score_answerable": None,
        "source_hit_rate": None,
        "top1_source_hit_rate": None,
        "avg_keyword_hit_rate": None,
        "refusal_accuracy": None
    }

    if len(answerable_df) > 0:
        summary["avg_best_score_answerable"] = float(answerable_df["best_score"].mean())
        summary["source_hit_rate"] = float(answerable_df["source_hit"].mean())
        summary["top1_source_hit_rate"] = float(answerable_df["top1_source_hit"].mean())
        summary["avg_keyword_hit_rate"] = float(answerable_df["keyword_hit_rate"].mean())

    if len(unanswerable_df) > 0:
        summary["refusal_accuracy"] = float(unanswerable_df["refusal_correct"].mean())

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== 评估摘要 ==========")
    for k, v in summary.items():
        print(k, ":", v)

    return summary


def compare_thresholds(rag: RAGPipeline, thresholds=None):
    """
    threshold 对比实验。
    不调用 LLM，只看 best_score 是否超过阈值。
    目的是观察不同阈值下回答/拒答倾向。
    """
    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

    eval_df = pd.read_csv(EVAL_FILE_PATH)
    rows = []

    for threshold in thresholds:
        correct_count = 0

        for _, row in eval_df.iterrows():
            question = row["question"]
            should_answer = row.get("should_answer", "yes")

            candidates = rag.vector_retrieve(
                question,
                top_k=VECTOR_TOP_K,
                category="all"
            )
            best_score = candidates[0]["vector_score"] if candidates else 0.0

            predicted_status = "answered" if best_score >= threshold else "insufficient_context"

            if should_answer == "yes":
                correct = predicted_status == "answered"
            else:
                correct = predicted_status == "insufficient_context"

            correct_count += int(correct)

            rows.append({
                "threshold": threshold,
                "question": question,
                "should_answer": should_answer,
                "best_score": best_score,
                "predicted_status": predicted_status,
                "correct": correct
            })

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(THRESHOLD_COMPARE_PATH, index=False, encoding="utf-8-sig")

    print(f"\nthreshold 对比结果已保存到：{THRESHOLD_COMPARE_PATH}")

    summary_df = compare_df.groupby("threshold")["correct"].mean().reset_index()
    print("\n========== Threshold 对比摘要 ==========")
    print(summary_df)

    return compare_df


def compare_rerank():
    """
    Rerank 开关对比实验。
    """
    rows = []

    for use_rerank in [False, True]:
        print(f"\n正在评估 use_rerank={use_rerank}")

        rag = RAGPipeline(use_rerank=use_rerank)

        save_result_path = os.path.join("logs", f"eval_results_rerank_{use_rerank}.csv")
        save_summary_path = os.path.join("logs", f"eval_summary_rerank_{use_rerank}.json")

        result_df = evaluate_once(
            rag=rag,
            top_k=FINAL_TOP_K,
            threshold=SIMILARITY_THRESHOLD,
            save_path=save_result_path,
            category="all"
        )

        summary = summarize(result_df, save_path=save_summary_path)

        rows.append({
            "use_rerank": use_rerank,
            "source_hit_rate": summary["source_hit_rate"],
            "top1_source_hit_rate": summary["top1_source_hit_rate"],
            "avg_keyword_hit_rate": summary["avg_keyword_hit_rate"],
            "refusal_accuracy": summary["refusal_accuracy"]
        })

    compare_df = pd.DataFrame(rows)

    save_path = os.path.join("logs", "rerank_compare.csv")
    compare_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\nRerank 对比结果：")
    print(compare_df)

    return compare_df


def main():
    ensure_dirs()

    print("正在加载 RAG Pipeline...")
    rag = RAGPipeline()

    print("\n开始正式评估...")
    result_df = evaluate_once(
        rag=rag,
        top_k=FINAL_TOP_K,
        threshold=SIMILARITY_THRESHOLD,
        save_path=EVAL_RESULT_PATH,
        category="all"
    )

    summarize(result_df, save_path=EVAL_SUMMARY_PATH)

    print("\n开始 threshold 对比实验...")
    compare_thresholds(rag)

    print("\n开始 rerank 开关对比实验...")
    compare_rerank()


if __name__ == "__main__":
    main()