import argparse
import csv
import json
import math
import re
import sys
import uuid
from collections.abc import Iterable as IterableABC
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from app.ingestion.pdf_to_markdown import PARENT_STORE_PATH
from app.retrieval.child_documents import (
    DEFAULT_CHILD_DOCS_PATH,
    filter_documents_by_access,
    load_child_documents,
    load_child_documents_from_qdrant,
)
from app.retrieval.context_builder import (
    build_main_secondary_context,
    retrieve_parent_docs_by_hybrid,
)
from app.retrieval.hybrid_retriever import SmartHybridRetriever
from app.security.access_control import build_access_filter

from app.config.settings import settings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def _doc_id_candidates(value: Any) -> set[str]:
    raw = str(value or "").strip().lower().replace("\\", "/")
    if not raw:
        return set()

    path_name = Path(raw).name

    candidates = {
        raw,
        path_name,
    }

    # Nếu có lúc expected là stem không có .md
    if "." in path_name:
        candidates.add(Path(path_name).stem)

    return {item for item in candidates if item}

def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, IterableABC):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _norm_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text), flags=re.UNICODE)


def _keyword_coverage(text: str, keywords: List[str]) -> Optional[float]:
    if not keywords:
        return None
    normalized = _normalize_text(text)
    hits = sum(1 for keyword in keywords if _normalize_text(keyword) in normalized)
    return hits / len(keywords)


def _ranking_metrics_from_flags(flags: List[bool], expected_count: int) -> Dict[str, float]:
    if expected_count <= 0:
        return {
            "hit": None,
            "precision": None,
            "recall": None,
            "mrr": None,
            "ndcg": None,
        }

    retrieved_count = len(flags)
    hit_count = sum(1 for flag in flags if flag)
    first_hit_rank = next((idx + 1 for idx, flag in enumerate(flags) if flag), None)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, flag in enumerate(flags, start=1)
        if flag
    )
    ideal_hits = min(expected_count, retrieved_count)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return {
        "hit": 1.0 if hit_count else 0.0,
        "precision": hit_count / retrieved_count if retrieved_count else 0.0,
        "recall": min(hit_count, expected_count) / expected_count,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def _parent_metrics(retrieved_parent_ids: List[str], expected_parent_ids: List[str]) -> Dict[str, float]:
    expected = {_norm_id(item) for item in expected_parent_ids}
    flags = [_norm_id(item) in expected for item in retrieved_parent_ids]
    return _ranking_metrics_from_flags(flags, len(expected))


def _doc_aliases_from_result(item: dict) -> List[str]:
    parent = item.get("parent", {}) or {}
    metadata = parent.get("metadata", {}) or {}

    aliases = [
        metadata.get("doc_id"),
        metadata.get("source"),
        metadata.get("source_md"),
        metadata.get("file_name"),
        metadata.get("filename"),
    ]
    return [str(alias).strip() for alias in aliases if str(alias or "").strip()]


def _doc_metrics(results: List[dict], expected_doc_ids: List[str]) -> Dict[str, float]:
    expected = set()
    for item in expected_doc_ids:
        expected.update(_doc_id_candidates(item))

    flags = []

    for item in results:
        aliases = set()
        for alias in _doc_aliases_from_result(item):
            aliases.update(_doc_id_candidates(alias))

        flags.append(bool(aliases & expected))

    return _ranking_metrics_from_flags(flags, len(expected_doc_ids))


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for idx_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[idx_b - 1] + 1)
            else:
                cur.append(max(prev[idx_b], cur[-1]))
        prev = cur
    return prev[-1]


def _token_f1(prediction: str, reference: str) -> Optional[float]:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return None

    ref_counts: Dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, reference: str) -> Optional[float]:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return None

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _sentence_support_rate(answer: str, context: str) -> Optional[float]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[\n.!?;]+", answer or "")
        if sentence.strip()
    ]
    if not sentences:
        return None

    context_tokens = set(_tokenize(context))
    if not context_tokens:
        return 0.0

    supported = 0
    checked = 0
    for sentence in sentences:
        tokens = _tokenize(sentence)
        if len(tokens) < 4:
            continue
        checked += 1
        overlap = sum(1 for token in tokens if token in context_tokens)
        ratio = overlap / len(tokens)
        if ratio >= 0.35 or overlap >= 5:
            supported += 1

    if checked == 0:
        return None
    return supported / checked


def load_eval_dataset(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    cases = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("question"):
                raise ValueError(f"Missing question at {path}:{line_no}")
            item.setdefault("id", f"case_{line_no:04d}")
            cases.append(item)

    return cases


def build_eval_retriever(
    child_k: int,
    user_id: str,
    role: str,
):
    from app.ingestion.indexing import CHILD_COLLECTION, QDRANT_PATH, build_child_vector_store

    client, child_vector_store = build_child_vector_store(
        collection_name=CHILD_COLLECTION,
        qdrant_path=QDRANT_PATH,
        recreate=False,
    )

    access_filter = build_access_filter(user_id=user_id, role=role)

    documents = load_child_documents(DEFAULT_CHILD_DOCS_PATH)
    if documents:
        documents = filter_documents_by_access(
            documents,
            user_id=user_id,
            role=role,
        )
    else:
        documents = load_child_documents_from_qdrant(
            client=client,
            collection_name=CHILD_COLLECTION,
            scroll_filter=access_filter,
        )

    retriever = SmartHybridRetriever(
        vector_store=child_vector_store,
        documents=documents,
        k=child_k,
        search_filter=access_filter,
    )

    return client, retriever


def run_retrieval_case(
    case: Dict[str, Any],
    child_k: int,
    max_parents: int,
    user_id: str,
    role: str,
    use_rerank: bool,
) -> Dict[str, Any]:
    query = case["question"]
    case_user_id = str(case.get("user_id") or user_id)
    case_role = str(case.get("role") or role)

    client, retriever = build_eval_retriever(
        child_k=child_k,
        user_id=case_user_id,
        role=case_role,
    )

    try:
        results = retrieve_parent_docs_by_hybrid(
            query=query,
            child_retriever=retriever,
            parent_store_path=str(PARENT_STORE_PATH),
            max_parents=max_parents,
            rank_constant=20,
            use_rerank=use_rerank,
        )
    finally:
        client.close()

    context_data = build_main_secondary_context(
        query=query,
        results=results,
        use_secondary_context=True,
        max_secondary_parents=2,
        secondary_ratio_threshold=0.4,
    )

    return {
        "results": results,
        "context_data": context_data,
    }


def evaluate_case(
    case: Dict[str, Any],
    child_k: int,
    max_parents: int,
    user_id: str,
    role: str,
    use_rerank: bool,
    run_agent: bool,
) -> Dict[str, Any]:
    retrieval_output = run_retrieval_case(
        case=case,
        child_k=child_k,
        max_parents=max_parents,
        user_id=user_id,
        role=role,
        use_rerank=use_rerank,
    )

    results = retrieval_output["results"]
    context_data = retrieval_output["context_data"]
    final_context = context_data.get("final_context", "") or ""

    retrieved_parent_ids = [item.get("parent_id", "") for item in results]
    retrieved_doc_aliases = [_doc_aliases_from_result(item) for item in results]

    expected_parent_ids = _as_list(
        case.get("expected_parent_ids") or case.get("gold_parent_ids")
    )
    expected_doc_ids = _as_list(
        case.get("expected_doc_ids") or case.get("gold_doc_ids")
    )
    expected_keywords = _as_list(case.get("expected_keywords"))

    parent_scores = _parent_metrics(retrieved_parent_ids, expected_parent_ids)
    doc_scores = _doc_metrics(results, expected_doc_ids)

    metrics = {
        "retrieved_parent_count": float(len(retrieved_parent_ids)),
        "context_chars": float(len(final_context)),
        "parent_hit_at_k": parent_scores["hit"],
        "parent_precision_at_k": parent_scores["precision"],
        "parent_recall_at_k": parent_scores["recall"],
        "parent_mrr_at_k": parent_scores["mrr"],
        "parent_ndcg_at_k": parent_scores["ndcg"],
        "doc_hit_at_k": doc_scores["hit"],
        "doc_precision_at_k": doc_scores["precision"],
        "doc_recall_at_k": doc_scores["recall"],
        "doc_mrr_at_k": doc_scores["mrr"],
        "doc_ndcg_at_k": doc_scores["ndcg"],
        "context_keyword_coverage": _keyword_coverage(final_context, expected_keywords),
    }

    answer = None
    if run_agent:
        from app.main_agent import run_agent as run_main_agent

        answer = run_main_agent(
            user_query=case["question"],
            thread_id=f"eval-{case.get('id')}-{uuid.uuid4()}",
            user_id=str(case.get("user_id") or user_id),
            role=str(case.get("role") or role),
        )

        expected_answer = str(case.get("expected_answer") or "").strip()
        metrics.update(
            {
                "answer_token_f1": _token_f1(answer, expected_answer)
                if expected_answer
                else None,
                "answer_rouge_l": _rouge_l(answer, expected_answer)
                if expected_answer
                else None,
                "answer_keyword_coverage": _keyword_coverage(answer, expected_keywords),
                "answer_grounded_sentence_rate": _sentence_support_rate(
                    answer,
                    final_context,
                ),
            }
        )

    return {
        "id": case.get("id"),
        "question": case["question"],
        "metrics": metrics,
        "retrieval": {
            "retrieved_parent_ids": retrieved_parent_ids,
            "retrieved_doc_aliases": retrieved_doc_aliases,
            "main_parent_id": (
                context_data.get("main_parent", {}) or {}
            ).get("parent_id"),
            "secondary_parent_ids": [
                item.get("parent_id")
                for item in context_data.get("secondary_parents", []) or []
            ],
        },
        "answer": answer,
    }


def summarize(case_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = sorted(
        {
            key
            for report in case_reports
            for key in (report.get("metrics") or {}).keys()
        }
    )

    summary = {
        "case_count": len(case_reports),
        "metrics": {},
    }

    for metric_name in metric_names:
        values = [
            report["metrics"].get(metric_name)
            for report in case_reports
            if isinstance(report.get("metrics", {}).get(metric_name), (int, float))
        ]
        if values:
            summary["metrics"][metric_name] = {
                "mean": mean(values),
                "count": len(values),
            }

    return summary


def write_csv_report(path: str | Path, case_reports: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = sorted(
        {
            key
            for report in case_reports
            for key in (report.get("metrics") or {}).keys()
        }
    )
    fieldnames = ["id", "question", "retrieved_parent_ids"] + metric_names

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for report in case_reports:
            row = {
                "id": report.get("id"),
                "question": report.get("question"),
                "retrieved_parent_ids": "|".join(
                    report.get("retrieval", {}).get("retrieved_parent_ids", [])
                ),
            }
            row.update(report.get("metrics") or {})
            writer.writerow(row)


def evaluate_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    csv_output_path: str | Path | None,
    child_k: int,
    max_parents: int,
    user_id: str,
    role: str,
    use_rerank: bool,
    run_agent: bool,
) -> Dict[str, Any]:
    cases = load_eval_dataset(dataset_path)
    case_reports = []

    for idx, case in enumerate(cases, start=1):
        print(f"[eval] {idx}/{len(cases)} {case.get('id')}: {case['question']}")
        case_reports.append(
            evaluate_case(
                case=case,
                child_k=child_k,
                max_parents=max_parents,
                user_id=user_id,
                role=role,
                use_rerank=use_rerank,
                run_agent=run_agent,
            )
        )

    report = {
        "dataset": str(dataset_path),
        "config": {
            "child_k": child_k,
            "max_parents": max_parents,
            "user_id": user_id,
            "role": role,
            "use_rerank": use_rerank,
            "run_agent": run_agent,
        },
        "summary": summarize(case_reports),
        "cases": case_reports,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if csv_output_path:
        write_csv_report(csv_output_path, case_reports)

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate this RAG pipeline.")
    parser.add_argument(
        "--dataset",
        default="data/eval/rag_eval_sample.jsonl",
        help="JSONL dataset with question and expected ids/answers.",
    )
    parser.add_argument(
        "--output",
        default="data/eval/rag_eval_report.json",
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--csv-output",
        default="data/eval/rag_eval_report.csv",
        help="Optional path to write a CSV report. Use empty string to skip.",
    )
    parser.add_argument("--child-k", type=int, default=settings.top_k_child)
    parser.add_argument("--max-parents", type=int, default=settings.top_k_parent)
    parser.add_argument("--user-id", default="anonymous")
    parser.add_argument(
        "--role",
        default="admin",
        help="Use admin by default so old indexed docs without visibility are included.",
    )
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Also call the full LangGraph agent and score generated answers.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    csv_output = args.csv_output.strip() if args.csv_output else None

    report = evaluate_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        csv_output_path=csv_output,
        child_k=args.child_k,
        max_parents=args.max_parents,
        user_id=args.user_id,
        role=args.role,
        use_rerank=not args.no_rerank,
        run_agent=args.run_agent,
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[eval] JSON report: {args.output}")
    if csv_output:
        print(f"[eval] CSV report: {csv_output}")


if __name__ == "__main__":
    main()
