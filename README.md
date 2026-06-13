# RAG LangGraph Assistant

A modular Retrieval-Augmented Generation (RAG) assistant built with **LangGraph**, **LangChain**, **FastAPI**, **Qdrant**, **Hugging Face embeddings**, and a local/custom LLM wrapper.

The project supports:

- Uploading PDF documents into a local vector database.
- Converting PDF content into Markdown.
- Building parent-child chunks for better document retrieval.
- Hybrid retrieval with dense + sparse embeddings.
- LangGraph-based agent orchestration.
- FastAPI endpoints for document upload and question answering.
- Optional market-data tool integration with `vnstock` for Vietnam stock price lookup.

---

## 1. Project Overview

This repository is designed as a local RAG chatbot/agent system.

The main workflow is:

```text
PDF document
    ↓
PDF → Markdown
    ↓
Parent-child chunking
    ↓
Qdrant local vector database
    ↓
Hybrid retrieval
    ↓
Reranking
    ↓
Final context construction
    ↓
LangGraph agent
    ↓
LLM-generated Vietnamese answer
```

The agent can be extended with additional tools, for example:

```text
RAG document retrieval
Stock price API via vnstock
```

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI, Uvicorn |
| Agent orchestration | LangGraph |
| LLM interface | Custom `QwenHFChat` wrapper |
| Retrieval | LangChain, LangChain Qdrant |
| Vector database | Qdrant local mode |
| Embeddings | Hugging Face / FastEmbed |
| Document parsing | PyMuPDF / Markdown pipeline |
| Stock data tool | `vnstock` |
| Package management | `uv` or `pip` |

---

## 3. Project Structure

```text
.
├── app/
│   ├── agents/
│   │   ├── graph.py          # Main LangGraph workflow
│   │   ├── nodes.py          # Graph nodes and routing logic
│   │   ├── state.py          # Graph state definitions
│   │   └── tools.py          # RAG tools and optional external tools
│   │
│   ├── chunking/
│   │   └── parent_child.py   # Parent-child chunking logic
│   │
│   ├── config/
│   │   ├── prompts.py        # Prompt templates
│   │   └── settings.py       # Project settings
│   │
│   ├── embeddings/
│   │   └── hf_embeddings.py  # Dense and sparse embeddings
│   │
│   ├── ingestion/
│   │   ├── indexing.py       # PDF indexing and Qdrant setup
│   │   └── pdf_to_markdown.py
│   │
│   ├── llms/
│   │   └── qwen_hf.py        # Custom LLM wrapper
│   │
│   ├── retrieval/
│   │   └── context_builder.py
│   │
│   ├── main_api.py           # FastAPI entrypoint
│   └── main_agent.py         # CLI/test entrypoint for LangGraph agent
│
├── data/
│   ├── markdown/             # Generated Markdown files
│   └── raw_pdfs/             # Optional raw PDF storage
│
├── qdrant_db/                # Local Qdrant database folder
├── parent_store/             # Parent chunk JSON store
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Requirements

Recommended environment:

- Python 3.10+
- Windows, Linux, or macOS
- `uv` package manager
- Internet connection for downloading models/dependencies
- Enough disk space for embedding models and local vector database

Check Python:

```bash
python --version
```

Install `uv` if you do not have it yet:

```bash
pip install uv
```

---

## 5. Installation

### 5.1. Clone the repository

```bash
git clone https://github.com/citun134/rag_langgragh.git
cd rag_langgragh
```

> Always run project commands from the repository root folder, not from inside `app/`.

Correct:

```bash
cd C:\RAG_PRO
```

Wrong:

```bash
cd C:\RAG_PRO\app
```

---

### 5.2. Create virtual environment with `uv`

```bash
uv venv
```

Activate it.

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

---

### 5.3. Install dependencies

Using `uv`:

```bash
uv pip install -r requirements.txt
```

---

## 6. Important Runtime Note

If you want to run the API using:

```bash
python -m app.main_api
```

make sure the bottom of `app/main_api.py` uses the correct module path:

```python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
```

This is important because the project is imported as a package from the repository root.

---

## 7. Running the Project

### 7.1. Run FastAPI server

From the repository root:

```bash
uv run python -m app.main_api
```

Alternative Uvicorn command:

```bash
uv run uvicorn app.main_api:app --host 0.0.0.0 --port 8000 --reload
```

After the server starts, open:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

### 7.2. Run the LangGraph agent directly

From the repository root:

```bash
uv run python -m app.main_agent
```

This runs the test query defined inside `app/main_agent.py`.

You can edit this line in `main_agent.py`:

```python
user_query = "xoài là gì"
```

Or, if the stock tool has been added:

```python
user_query = "Lấy giá cổ phiếu FPT ngày 2024-05-24"
```

---

## 8. API Usage

### 8.1. Upload PDF into the local vector database

Endpoint:

```http
POST /upload-pdf
```

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf" ^
  -F "file=@C:\path\to\your_document.pdf"
```

On Linux/macOS:

```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf" \
  -F "file=@/path/to/your_document.pdf"
```

Successful response:

```json
{
  "status": "success",
  "message": "Đã lưu 'your_document.pdf' vào DB thành công.",
  "filename": "your_document.pdf"
}
```

---

### 8.2. Ask a question

Endpoint:

```http
POST /query
```

Example with `curl` on Windows PowerShell/CMD:

```bash
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Tóm tắt tài liệu vừa upload\",\"thread_id\":\"test-thread\"}"
```

On Linux/macOS:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Tóm tắt tài liệu vừa upload","thread_id":"test-thread"}'
```

Example response:

```json
{
  "answer": "Nội dung trả lời từ agent...",
  "sources": []
}
```

---

## 9. Vietnam Stock Price Tool with `vnstock`

If you added the `get_vietnam_stock_price` tool inside `app/agents/tools.py` and registered it in the LangGraph `ToolNode`, you can ask questions like:

```text
Lấy giá cổ phiếu FPT ngày 2024-05-24
```

or:

```text
Cho tôi OHLCV của mã HPG ngày 2024-05-24
```

Suggested tool behavior:

```text
User question
    ↓
LangGraph orchestrator
    ↓
get_vietnam_stock_price tool
    ↓
JSON result from vnstock
    ↓
LLM formats the answer in Vietnamese
```

The returned fields should include:

```json
{
  "symbol": "FPT",
  "source": "KBS",
  "requested_date": "2024-05-24",
  "trading_date": "2024-05-24",
  "is_exact_match": true,
  "mode": "previous",
  "open": 0,
  "high": 0,
  "low": 0,
  "close": 0,
  "volume": 0
}
```

> The numbers above are only an example format. Real values are returned by `vnstock`.

---

## 10. Indexing Documents Manually

You can also run the indexing pipeline manually:

```bash
uv run python -m app.ingestion.indexing
```

This will call:

```python
index_documents_semantic_parent_child()
```

and build/update the local Qdrant database.

Default storage paths:

```text
qdrant_db/
parent_store/
data/markdown/
```

---

## 11. Evaluate RAG

This repo includes a lightweight offline evaluator for the current parent-child hybrid RAG pipeline.

Run retrieval/context evaluation:

```bash
uv run python -m app.evaluation.rag_evaluate --dataset data/eval/rag_eval_sample.jsonl
```

Fast smoke test without loading the reranker:

```bash
uv run python -m app.evaluation.rag_evaluate --dataset data/eval/rag_eval_sample.jsonl --no-rerank
```

Run full evaluation, including the LangGraph agent answer:

```bash
uv run python -m app.evaluation.rag_evaluate --dataset data/eval/rag_eval_sample.jsonl --run-agent
```

Reports are written to:

```text
data/eval/rag_eval_report.json
data/eval/rag_eval_report.csv
```

Create your own JSONL dataset with fields such as `question`, `expected_parent_ids`, `expected_doc_ids`, `expected_keywords`, and optional `expected_answer`.

---


## 12. License

No license has been specified yet.

If this is an open-source project, consider adding a `LICENSE` file such as:

- MIT
- Apache-2.0
- BSD-3-Clause

---

## 13. Author

Maintained by `citun134`.
