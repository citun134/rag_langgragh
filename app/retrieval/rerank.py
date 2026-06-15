import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config.settings import settings

_model = None
_tokenizer = None

def _get_reranker():
    global _model, _tokenizer

    if _model is None:
        print("Loading reranker model...")
        model_name = settings.reranker_model

        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )

        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()

        if torch.cuda.is_available():
            _model = _model.cuda()

        print("✓ Reranker loaded.")

    return _tokenizer, _model


def rerank_child_docs(
    query: str,
    child_docs: list,
    top_k: int | None = None,
    batch_size: int = 16,
) -> list:
    if not child_docs:
        return child_docs

    tokenizer, model = _get_reranker()
    device = next(model.parameters()).device

    all_scores = []

    for start in range(0, len(child_docs), batch_size):
        batch_docs = child_docs[start:start + batch_size]
        pairs = [(query, doc.page_content) for doc in batch_docs]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1).detach().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        all_scores.extend(scores)

    scored = sorted(
        zip(all_scores, child_docs),
        key=lambda x: x[0],
        reverse=True,
    )

    selected = scored[:top_k] if top_k else scored

    results = []
    for score, doc in selected:
        doc.metadata["rerank_score"] = float(score)
        results.append(doc)

    return results