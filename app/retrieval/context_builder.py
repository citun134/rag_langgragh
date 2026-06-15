from collections import defaultdict
import os
import json
import re
import math
from app.retrieval.rerank import rerank_child_docs

RERANK_CANDIDATE_MULTIPLIER = 4
RERANK_MIN_CANDIDATES = 12

def load_parent_doc(parent_id: str, parent_store_path: str):
    path = os.path.join(parent_store_path, f"{parent_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str):
    text = _normalize_text(text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _header_text(metadata: dict) -> str:
    parts = []
    for k in ["H1", "H2", "H3", "header_prefix"]:
        v = metadata.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts)


def _build_query_idf_weights(query: str, child_docs: list):
    """
    Tính IDF nhẹ trên tập candidate child hiện tại.
    Query token nào hiếm hơn trong candidate set thì quan trọng hơn.
    """
    q_tokens = list(dict.fromkeys(_tokenize(query)))
    if not q_tokens:
        return {}

    N = max(len(child_docs), 1)
    df = {tok: 0 for tok in q_tokens}

    for doc in child_docs:
        text = (doc.page_content or "") + "\n" + _header_text(doc.metadata or {})
        tokens = set(_tokenize(text))
        for tok in q_tokens:
            if tok in tokens:
                df[tok] += 1

    idf = {}
    for tok in q_tokens:
        # smoothed idf
        idf[tok] = math.log((N + 1) / (df[tok] + 1)) + 1.0
    return idf


def _weighted_token_overlap_score(query: str, text: str, idf_weights: dict) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0

    matched = q_tokens & t_tokens
    num = sum(idf_weights.get(tok, 1.0) for tok in matched)
    den = sum(idf_weights.get(tok, 1.0) for tok in q_tokens)
    if den <= 0:
        return 0.0
    return num / den


def _extract_query_phrases(query: str, min_n: int = 2, max_n: int = 4):
    """
    Sinh phrase động từ chính query, không hardcode.
    Chỉ giữ phrase đủ dài để tránh nhiễu.
    """
    tokens = _tokenize(query)
    phrases = []
    n_tokens = len(tokens)

    for n in range(min_n, min(max_n, n_tokens) + 1):
        for i in range(n_tokens - n + 1):
            phrase = " ".join(tokens[i:i + n]).strip()
            if len(phrase) >= 8:
                phrases.append(phrase)

    # phrase dài ưu tiên trước
    phrases = sorted(set(phrases), key=lambda x: (-len(x), x))
    return phrases


def _dynamic_phrase_match_score(query: str, text: str) -> float:
    """
    Bonus cho các phrase được sinh từ query xuất hiện nguyên cụm trong text.
    Không cố định phrase theo domain nào.
    """
    t = _normalize_text(text)
    phrases = _extract_query_phrases(query)

    score = 0.0
    for phrase in phrases[:12]:  # giới hạn để đỡ quá mạnh
        if phrase in t:
            # phrase dài hơn thì điểm cao hơn chút
            words = len(phrase.split())
            score += min(0.04 * words, 0.16)

    return min(score, 0.30)


def _extract_special_tokens(text: str):
    """
    Lấy các token 'đặc biệt' như:
    - có số: HS2028, 05/01/2026, 2028
    - mã/cụm dạng viết tắt có dấu / -
    """
    text = text or ""
    patterns = re.findall(r"[A-Za-zÀ-ỹ0-9_./-]*\d+[A-Za-zÀ-ỹ0-9_./-]*", text)
    return set(p.lower() for p in patterns if len(p) >= 2)


def _special_token_match_score(query: str, text: str) -> float:
    q_special = _extract_special_tokens(query)
    t_special = _extract_special_tokens(text)
    if not q_special or not t_special:
        return 0.0

    overlap = q_special & t_special
    return min(0.12 * len(overlap), 0.24)


def retrieve_parent_docs_by_hybrid(
        query: str,
        child_retriever,
        parent_store_path: str,
        max_parents: int = 4,
        rank_constant: int = 20,
        max_children_per_parent: int = 3,
        use_rerank: bool = True,  # ← thêm tham số
):
    """
    Generic version:
    - không hardcode phrase theo domain
    - không hardcode table cues theo country/query cụ thể
    - dùng weighted overlap + dynamic phrase + special token match
    """

    child_docs = child_retriever.invoke(query)

    # ← THÊM BƯỚC RERANK NÀY
    if use_rerank and len(child_docs) > 3:
        RERANK_TOP_K = 25
        child_docs = rerank_child_docs(
            query=query,
            child_docs=child_docs,
            top_k=RERANK_TOP_K,
            batch_size=16,
        )

    idf_weights = _build_query_idf_weights(query, child_docs)

    parent_buckets = defaultdict(lambda: {
        "children": [],
        "seen_child_ids": set(),
    })

    for rank, child_doc in enumerate(child_docs, 1):
        metadata = dict(child_doc.metadata or {})
        parent_id = metadata.get("parent_id")
        child_id = metadata.get("child_id")

        if not parent_id:
            continue

        if child_id and child_id in parent_buckets[parent_id]["seen_child_ids"]:
            continue

        child_text = child_doc.page_content or ""
        header_text = _header_text(metadata)
        combined_text = header_text + "\n" + child_text

        # 1) điểm từ retriever gốc
        base_rank_score = 1.0 / (rank_constant + rank)

        # 2) lexical match có trọng số hiếm-phổ biến
        lexical_score = _weighted_token_overlap_score(query, child_text, idf_weights)

        # 3) header/title match
        header_score = _weighted_token_overlap_score(query, header_text, idf_weights)

        # 4) phrase động từ query
        phrase_score = _dynamic_phrase_match_score(query, combined_text)

        # 5) số / ngày / mã / token đặc biệt
        special_score = _special_token_match_score(query, combined_text)

        # child score tổng quát
        final_child_score = (
                0.30 * base_rank_score +
                0.30 * lexical_score +
                0.25 * header_score +
                0.10 * phrase_score +
                0.05 * special_score
        )

        if child_id:
            parent_buckets[parent_id]["seen_child_ids"].add(child_id)

        parent_buckets[parent_id]["children"].append({
            "rank": rank,
            "rank_score": final_child_score,
            "base_rank_score": base_rank_score,
            "lexical_score": lexical_score,
            "header_score": header_score,
            "phrase_score": phrase_score,
            "special_score": special_score,
            "page_content": child_text,
            "metadata": metadata
        })

    ranked_parents = []

    for parent_id, info in parent_buckets.items():
        # chỉ giữ top child mạnh nhất mỗi parent
        info["children"] = sorted(
            info["children"],
            key=lambda x: x["rank_score"],
            reverse=True
        )[:max_children_per_parent]

        best_child_score = max((c["rank_score"] for c in info["children"]), default=0.0)
        mean_child_score = (
                sum(c["rank_score"] for c in info["children"]) / max(len(info["children"]), 1)
        )
        coverage_bonus = min(0.03 * len(info["children"]), 0.09)

        # parent score: ưu tiên best child hơn là cộng dồn mù quáng
        parent_score = (
                0.60 * best_child_score +
                0.30 * mean_child_score +
                0.10 * coverage_bonus
        )

        ranked_parents.append((parent_id, parent_score, best_child_score, info))

    ranked_parents.sort(key=lambda x: (x[1], x[2]), reverse=True)

    results = []
    for parent_id, parent_score, best_child_score, info in ranked_parents[:max_parents]:
        parent_data = load_parent_doc(parent_id, parent_store_path)
        if parent_data is None:
            continue

        results.append({
            "parent_id": parent_id,
            "parent_rank_score_sum": parent_score,
            "parent_best_rank_score": best_child_score,
            "parent": parent_data,
            "children": info["children"]
        })

    return results


def build_main_secondary_context(
        query: str,
        results: list,
        use_secondary_context: bool = True,
        max_secondary_parents: int = 2,
        secondary_ratio_threshold: float = 0.4,
        max_parent_chars: int = 1400,  # ← giảm từ 2500 → 1400
        max_child_chars: int = 400,  # ← giảm từ 600 → 400
):
    """
    Version đơn giản:
    - KHÔNG rerank lại results
    - tin vào thứ tự do retriever trả về
    - results[0] = main context
    - results[1:] = secondary context nếu bật
    """

    # Thêm: detect summary query → relax threshold hơn nữa
    q_lower = (query or "").lower()
    is_summary = any(m in q_lower for m in ["tóm tắt", "tổng quan", "kết luận", "summary"])
    is_enum = any(m in q_lower for m in ["gồm gì", "gồm những", "liệt kê", "những gì", "nêu ra"])

    if is_summary:
        secondary_ratio_threshold = 0.25
        max_secondary_parents = 3
        max_parent_chars = 2000  # summary cần nhiều hơn một chút
        max_child_chars = 500
    elif is_enum:
        max_parent_chars = 1800  # enum cần đủ để có danh sách
        max_child_chars = 500
        max_secondary_parents = 2
        secondary_ratio_threshold = 0.35

    if not results:
        return {
            "main_parent": None,
            "secondary_parents": [],
            "main_context": "",
            "secondary_context": "",
            "final_context": "",
        }

    def get_parent_score(item):
        if "parent_rank_score_sum" in item:
            return float(item["parent_rank_score_sum"])
        if "parent_final_score" in item:
            return float(item["parent_final_score"])
        if "parent_score_sum" in item:
            return float(item["parent_score_sum"])
        return 0.0

    def format_parent_block(item, tag):
        parent = item.get("parent", {}) or {}
        children = item.get("children", []) or []

        parent_id = item.get("parent_id", "unknown_parent")
        parent_score = get_parent_score(item)
        parent_text = (parent.get("page_content", "") or "")[:max_parent_chars]

        lines = [
            f"[{tag}]",
            f"parent_id: {parent_id}",
            f"score: {parent_score:.6f}",
            "",
            "PARENT CONTENT:",
            parent_text,
        ]

        # if children:
        #     best_child = children[0]
        #     child_text = (best_child.get("page_content", "") or "")[:max_child_chars]
        #     lines.extend([
        #         "",
        #         "BEST MATCHED CHILD:",
        #         child_text,
        #     ])

        # SAU: lấy tất cả children (đã giới hạn max_children_per_parent=3)
        if children:
            child_texts = []
            for c in children[:3]:
                ct = (c.get("page_content", "") or "").strip()[:max_child_chars]
                if ct:
                    child_texts.append(ct)
            lines.extend(["", "MATCHED CHILDREN:", "\n---\n".join(child_texts)])

        return "\n".join(lines)

    # GIỮ NGUYÊN thứ tự retriever đã trả
    ranked_results = results

    main_item = ranked_results[0]
    main_score = get_parent_score(main_item)

    secondary_items = []
    if use_secondary_context:
        for item in ranked_results[1:]:
            score = get_parent_score(item)
            ratio = score / (main_score + 1e-12)

            if ratio >= secondary_ratio_threshold:
                secondary_items.append(item)

            if len(secondary_items) >= max_secondary_parents:
                break

    main_context = format_parent_block(main_item, "MAIN CONTEXT")

    secondary_context = ""
    if secondary_items:
        blocks = []
        for idx, item in enumerate(secondary_items, 1):
            blocks.append(format_parent_block(item, f"SECONDARY CONTEXT {idx}"))
        secondary_context = "\n\n".join(blocks)

    final_context = main_context
    if use_secondary_context and secondary_context.strip():
        final_context = main_context + "\n\n" + secondary_context

    return {
        "main_parent": main_item,
        "secondary_parents": secondary_items,
        "main_context": main_context,
        "secondary_context": secondary_context,
        "final_context": final_context,
    }