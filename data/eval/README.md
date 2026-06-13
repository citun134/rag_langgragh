# RAG evaluation

Run retrieval/context evaluation:

```powershell
python -m app.evaluation.rag_evaluate --dataset data/eval/rag_eval_sample.jsonl
```

Fast smoke test without loading the reranker:

```powershell
python -m app.evaluation.rag_evaluate --dataset data/eval/rag_eval_sample.jsonl --no-rerank
```

Run full evaluation, including the LangGraph agent answer:

```powershell
python -m app.evaluation.rag_evaluate --dataset data/eval/rag_eval_sample.jsonl --run-agent
```

Dataset format is JSONL, one question per line:

```json
{
  "id": "case_id",
  "question": "User question",
  "expected_parent_ids": ["parent_id_1"],
  "expected_doc_ids": ["optional_doc_id_or_filename.pdf"],
  "expected_keywords": ["keyword 1", "keyword 2"],
  "expected_answer": "Optional reference answer",
  "user_id": "anonymous",
  "role": "admin"
}
```

Important metrics:

- `parent_hit_at_k`, `parent_recall_at_k`, `parent_mrr_at_k`: whether the expected parent chunks were retrieved.
- `doc_hit_at_k`, `doc_recall_at_k`: whether the expected source document was retrieved.
- `context_keyword_coverage`: whether the retrieved context contains expected keywords.
- `answer_token_f1`, `answer_rouge_l`, `answer_keyword_coverage`: answer metrics, only when `--run-agent` is used.
- `answer_grounded_sentence_rate`: simple groundedness estimate, only when `--run-agent` is used.

The default role is `admin` because older indexed documents in this repo may not have access metadata.
