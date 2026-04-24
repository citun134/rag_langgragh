import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_model = None
_tokenizer = None

def _get_reranker():
    global _model, _tokenizer
    if _model is None:
        print("Loading reranker model...")
        model_name = "AITeamVN/Vietnamese_Reranker"
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # ← use_fast=False tránh bug
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
        print("✓ Reranker loaded.")
    return _tokenizer, _model


def rerank_child_docs(query: str, child_docs: list, top_k: int = None) -> list:
    if not child_docs:
        return child_docs

    tokenizer, model = _get_reranker()
    device = next(model.parameters()).device

    pairs = [(query, doc.page_content) for doc in child_docs]

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
        scores = outputs.logits.squeeze(-1).cpu().tolist()

    if isinstance(scores, float):
        scores = [scores]

    scored = sorted(
        zip(scores, child_docs),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, doc in (scored[:top_k] if top_k else scored):
        doc.metadata["rerank_score"] = float(score)
        results.append(doc)

    return results