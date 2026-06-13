import re
import uuid
import json
import pandas as pd
from vnstock import Quote
from langchain_core.tools import tool
from datetime import datetime, timedelta


# =========================================================
# RETRIEVAL SETUP
# =========================================================
from app.ingestion.indexing import (
    build_child_vector_store,
    QDRANT_PATH,
    CHILD_COLLECTION,
)
from app.ingestion.pdf_to_markdown import PARENT_STORE_PATH
from app.retrieval.context_builder import (
    retrieve_parent_docs_by_hybrid,
    build_main_secondary_context,
)
from app.security.access_control import build_access_filter

# def build_retriever(child_vector_store, k: int = 20):
#     """Wrap vector store thành retriever có .invoke()"""
#     return child_vector_store.as_retriever(search_kwargs={"k": k})
def build_retriever(child_vector_store, k: int = 20, access_filter=None):
    """
    Wrap vector store as retriever with access filter.
    """

    search_kwargs = {"k": k}

    if access_filter is not None:
        search_kwargs["filter"] = access_filter

    return child_vector_store.as_retriever(search_kwargs=search_kwargs)

# =========================================================
# KHỞI TẠO RETRIEVER TOÀN CỤC (dùng trong tools.py)
# =========================================================
# def get_child_hybrid_retriever(k: int = 20):
#     client, child_vector_store = build_child_vector_store(
#         collection_name=CHILD_COLLECTION,
#         qdrant_path=QDRANT_PATH,
#         recreate=False,
#     )
#     retriever = build_retriever(child_vector_store, k=k)
#     return client, retriever
def get_child_hybrid_retriever(
    k: int = 20,
    user_id: str = "anonymous",
    role: str = "employee",
):
    client, child_vector_store = build_child_vector_store(
        collection_name=CHILD_COLLECTION,
        qdrant_path=QDRANT_PATH,
        recreate=False,
    )

    access_filter = build_access_filter(
        user_id=user_id,
        role=role,
    )

    retriever = build_retriever(
        child_vector_store=child_vector_store,
        k=k,
        access_filter=access_filter,
    )

    return client, retriever


# =========================================================
# HELPERS
# =========================================================
def _to_tool_call(parsed: dict):
    return {
        "name": parsed["tool_name"],
        "args": parsed.get("arguments", {}) or {},
        "id": str(uuid.uuid4()),
        "type": "tool_call",
    }

def _make_tool_call(name: str, args: dict):
    return {
        "name": name,
        "args": args or {},
        "id": str(uuid.uuid4()),
        "type": "tool_call",
    }

@tool
def retrieve_hybrid_context(
    query: str,
    user_id: str = "anonymous",
    role: str = "employee",
    max_parents: int = 3,
    rank_constant: int = 20,
    use_secondary_context: bool = True,
    max_secondary_parents: int = 1,
    secondary_ratio_threshold: float = 0.55,
) -> str:
    """Retrieve relevant document context using hybrid search for a given query."""
    q_lower = query.lower()

    is_summary = any(m in q_lower for m in ["tóm tắt", "tổng quan", "kết luận"])
    is_enum = any(m in q_lower for m in [
        "gồm gì", "gồm những", "bao gồm gì", "liệt kê", "những gì", "nêu ra", "gồm có"
    ])

    if is_summary:
        max_parents = max(max_parents, 5)
        max_secondary_parents = 3
        secondary_ratio_threshold = 0.25
    elif is_enum:  # ← THÊM khối này
        max_parents = max(max_parents, 4)
        max_secondary_parents = 2
        secondary_ratio_threshold = 0.35

    # client, child_hybrid_retriever = get_child_hybrid_retriever(k=20)
    client, child_hybrid_retriever = get_child_hybrid_retriever(
        k=20,
        user_id=user_id,
        role=role,
    )

    try:
        results = retrieve_parent_docs_by_hybrid(
            query=query,
            child_retriever=child_hybrid_retriever,
            parent_store_path=PARENT_STORE_PATH,
            max_parents=max_parents,
            rank_constant=rank_constant,
        )
    finally:
        client.close()

    context_data = build_main_secondary_context(
        query=query,
        results=results,
        use_secondary_context=use_secondary_context,
        max_secondary_parents=max_secondary_parents,
        secondary_ratio_threshold=secondary_ratio_threshold,
    )

    # Optional: keep only lightweight metadata to avoid huge tool payload
    compact_results = []
    for item in results:
        compact_results.append({
            "parent_id": item.get("parent_id"),
            "score": item.get("score"),
            "child_preview": str(item.get("best_child_text", ""))[:500]
        })

    payload = {
        "query": query,
        "main_context": context_data.get("main_context", ""),
        "secondary_context": context_data.get("secondary_context", ""),
        "final_context": context_data.get("final_context", ""),
        "results_meta": compact_results,
    }

    return json.dumps(payload, ensure_ascii=False)


# =========================================================
# KHỞI TẠO VNSTOCK
# =========================================================

def _parse_date_yyyy_mm_dd(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            "Ngày phải có định dạng YYYY-MM-DD, ví dụ: 2024-05-24"
        ) from exc


def _normalize_vnstock_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "time" not in df.columns:
        raise ValueError(
            f"Dữ liệu vnstock không có cột 'time'. Columns hiện có: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["date"] = df["time"].dt.date
    df = df.sort_values("time").reset_index(drop=True)

    return df


def _safe_cell(row, col: str):
    value = row.get(col)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@tool
def get_vietnam_stock_price(
    symbol: str,
    date: str,
    source: str = "KBS",
    mode: str = "previous",
) -> str:
    """
    Lấy giá OHLCV của mã cổ phiếu/chỉ số Việt Nam theo ngày.

    Args:
        symbol: Mã cổ phiếu/chỉ số, ví dụ FPT, VNM, HPG, VNINDEX.
        date: Ngày cần lấy giá, định dạng YYYY-MM-DD.
        source: Nguồn dữ liệu, nên dùng KBS hoặc VCI.
        mode:
            - exact: chỉ lấy đúng ngày yêu cầu.
            - previous: nếu ngày yêu cầu không có giao dịch, lấy phiên gần nhất trước đó.
            - nearest: lấy phiên giao dịch gần nhất quanh ngày yêu cầu.
    """
    symbol = symbol.upper().strip()
    source = source.upper().strip()
    mode = mode.lower().strip()

    if source not in {"KBS", "VCI"}:
        raise ValueError("source chỉ được là 'KBS' hoặc 'VCI'.")

    if mode not in {"exact", "previous", "nearest"}:
        raise ValueError("mode chỉ được là 'exact', 'previous' hoặc 'nearest'.")

    target_dt = _parse_date_yyyy_mm_dd(date)
    target_date = target_dt.date()

    # Lùi vài ngày để xử lý thứ 7, chủ nhật, ngày nghỉ lễ.
    start_date = (target_dt - timedelta(days=14)).strftime("%Y-%m-%d")
    end_date = (target_dt + timedelta(days=3)).strftime("%Y-%m-%d")

    quote = Quote(symbol=symbol, source=source)
    df = quote.history(
        start=start_date,
        end=end_date,
        interval="1D",
    )

    df = _normalize_vnstock_df(df)

    if df.empty:
        payload = {
            "type": "stock_price",
            "ok": False,
            "error": f"Không tìm thấy dữ liệu cho mã {symbol} quanh ngày {date}.",
            "symbol": symbol,
            "requested_date": date,
            "source": source,
        }
        return json.dumps(payload, ensure_ascii=False)

    exact_df = df[df["date"] == target_date]

    if not exact_df.empty:
        row = exact_df.iloc[-1]
        is_exact_match = True
    else:
        if mode == "exact":
            payload = {
                "type": "stock_price",
                "ok": False,
                "error": f"Ngày {date} không có dữ liệu giao dịch cho mã {symbol}.",
                "symbol": symbol,
                "requested_date": date,
                "source": source,
            }
            return json.dumps(payload, ensure_ascii=False)

        if mode == "previous":
            candidates = df[df["date"] <= target_date]
            if candidates.empty:
                payload = {
                    "type": "stock_price",
                    "ok": False,
                    "error": f"Không tìm thấy phiên giao dịch trước ngày {date}.",
                    "symbol": symbol,
                    "requested_date": date,
                    "source": source,
                }
                return json.dumps(payload, ensure_ascii=False)
            row = candidates.iloc[-1]

        else:
            df = df.copy()
            df["distance"] = df["date"].apply(lambda d: abs((d - target_date).days))
            row = df.sort_values(["distance", "date"]).iloc[0]

        is_exact_match = False

    payload = {
        "type": "stock_price",
        "ok": True,
        "symbol": symbol,
        "source": source,
        "requested_date": date,
        "trading_date": str(row["date"]),
        "is_exact_match": is_exact_match,
        "mode": mode,
        "open": _safe_cell(row, "open"),
        "high": _safe_cell(row, "high"),
        "low": _safe_cell(row, "low"),
        "close": _safe_cell(row, "close"),
        "volume": _safe_cell(row, "volume"),
    }

    return json.dumps(payload, ensure_ascii=False)

# =========================================================
# TEST NHANH VNSTOCK TOOL
# =========================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Test get_vietnam_stock_price tool with vnstock"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="FPT",
        help="Mã cổ phiếu, ví dụ: FPT, VNM, HPG",
    )

    parser.add_argument(
        "--date",
        type=str,
        default="2024-05-24",
        help="Ngày cần lấy giá, định dạng YYYY-MM-DD",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="KBS",
        choices=["KBS", "VCI"],
        help="Nguồn dữ liệu vnstock",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="previous",
        choices=["exact", "previous", "nearest"],
        help="Cách xử lý nếu ngày yêu cầu không có giao dịch",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TEST GET VIETNAM STOCK PRICE")
    print("=" * 80)
    print(f"Symbol : {args.symbol}")
    print(f"Date   : {args.date}")
    print(f"Source : {args.source}")
    print(f"Mode   : {args.mode}")
    print("-" * 80)

    try:
        # Vì get_vietnam_stock_price đang được decorate bằng @tool
        # nên gọi bằng .invoke({...})
        result_str = get_vietnam_stock_price.invoke(
            {
                "symbol": args.symbol,
                "date": args.date,
                "source": args.source,
                "mode": args.mode,
            }
        )

        print("RAW RESULT:")
        print(result_str)

        print("\nFORMATTED RESULT:")
        result = json.loads(result_str)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result.get("ok"):
            print("\nSUMMARY:")
            print(f"Mã cổ phiếu       : {result.get('symbol')}")
            print(f"Ngày yêu cầu      : {result.get('requested_date')}")
            print(f"Ngày giao dịch    : {result.get('trading_date')}")
            print(f"Khớp đúng ngày    : {result.get('is_exact_match')}")
            print(f"Open              : {result.get('open')}")
            print(f"High              : {result.get('high')}")
            print(f"Low               : {result.get('low')}")
            print(f"Close             : {result.get('close')}")
            print(f"Volume            : {result.get('volume')}")
        else:
            print("\nERROR:")
            print(result.get("error"))

    except Exception as e:
        print("\nEXCEPTION:")
        print(type(e).__name__, str(e))