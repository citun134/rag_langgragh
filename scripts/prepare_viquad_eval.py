import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_doc_id(title: str, context: str) -> str:
    """
    Tạo doc_id ổn định theo nội dung context.
    Dùng hash để tránh trùng title.
    """
    raw = normalize_text(title) + "\n" + normalize_text(context)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"viquad_ctx_{h}"


def safe_markdown_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def get_first_answer(row):
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    if not texts:
        return ""
    return str(texts[0]).strip()


def get_plausible_answer(row):
    plausible = row.get("plausible_answers") or {}
    texts = plausible.get("text") or []
    if not texts:
        return ""
    return str(texts[0]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-contexts", type=int, default=500)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--include-unanswerable", action="store_true")
    parser.add_argument("--docs-dir", default="data/eval/viquad_docs")
    parser.add_argument("--eval-output", default="data/eval/viquad_eval.jsonl")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    eval_output = Path(args.eval_output)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    print("[load] Loading UIT-ViQuAD2.0 from Hugging Face...")
    ds = load_dataset("taidng/UIT-ViQuAD2.0")
    rows = ds[args.split]

    context_to_doc = {}
    eval_cases = []
    written_doc_count = 0

    for row in tqdm(rows, desc="Preparing ViQuAD"):
        title = normalize_text(row.get("title", ""))
        context = normalize_text(row.get("context", ""))
        question = normalize_text(row.get("question", ""))

        if not context or not question:
            continue

        is_impossible = bool(row.get("is_impossible", False))

        # Giai đoạn đầu nên bỏ câu hỏi không có đáp án.
        # Sau khi retrieval ổn rồi mới bật --include-unanswerable để test refusal.
        if is_impossible and not args.include_unanswerable:
            continue

        answer_text = get_first_answer(row)

        if not is_impossible and not answer_text:
            continue

        doc_id = make_doc_id(title, context)
        file_name = f"{doc_id}.md"

        # Giới hạn số context để test nhẹ.
        if doc_id not in context_to_doc:
            if written_doc_count >= args.max_contexts:
                continue

            context_to_doc[doc_id] = file_name
            written_doc_count += 1

            md_path = docs_dir / file_name
            md_content = f"""# {safe_markdown_text(title)}

{safe_markdown_text(context)}
"""
            md_path.write_text(md_content, encoding="utf-8")

        # Nếu là câu trả lời được, expected_answer là answer thật.
        if not is_impossible:
            expected_answer = answer_text
            expected_keywords = [answer_text]
        else:
            # Case khó: context KHÔNG chứa đáp án thật.
            # Ta muốn agent trả lời kiểu: không có thông tin trong tài liệu.
            plausible_answer = get_plausible_answer(row)
            expected_answer = "Không có thông tin trong tài liệu được cung cấp."
            expected_keywords = []
            if plausible_answer:
                expected_keywords.append(plausible_answer)

        case = {
            "id": str(row.get("id") or f"viquad_case_{len(eval_cases)+1}"),
            "question": question,
            "expected_doc_ids": [file_name],
            "expected_keywords": expected_keywords,
            "expected_answer": expected_answer,
            "source_dataset": "taidng/UIT-ViQuAD2.0",
            "title": title,
            "is_impossible": is_impossible,
        }

        eval_cases.append(case)

        if len(eval_cases) >= args.max_cases:
            break

    with eval_output.open("w", encoding="utf-8") as f:
        for case in eval_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Docs written: {written_doc_count}")
    print(f"Eval cases:   {len(eval_cases)}")
    print(f"Docs dir:     {docs_dir}")
    print(f"Eval file:    {eval_output}")


if __name__ == "__main__":
    main()