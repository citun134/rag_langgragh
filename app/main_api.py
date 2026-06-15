from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import tempfile
import os
import shutil
import traceback
from typing import Optional
from app.security.access_control import build_document_metadata
import json

# ============================================================
# Import từ code gốc của bạn — chỉnh lại đường dẫn nếu cần
# ============================================================
from app.ingestion.indexing import add_pdf_to_db   # hàm add_pdf_to_db(pdf_path)
from app.agents.graph import agent_graph  # noqa: E402

# Nếu graph nằm ở file khác, ví dụ:
# from graph import app as graph
# from main import graph, add_pdf_to_db
# ============================================================

app = FastAPI(
    title="RAG API",
    description="API để upload PDF vào DB và query RAG chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"   # dùng để giữ lịch sử chat nếu graph hỗ trợ
    user_id: str = "anonymous"
    role: str = "employee"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


# ─────────────────────────────────────────────
# API 1: Upload PDF → lưu vào DB
# ─────────────────────────────────────────────

@app.post("/upload-pdf", summary="Upload PDF và lưu vào vector DB")
async def upload_pdf(
    file: UploadFile = File(...),
    visibility: str = Form("private"),
    owner_user_id: str = Form(...),
    doc_id: Optional[str] = Form(None),
):
    """
    Upload PDF with access metadata.

    visibility:
    - public
    - internal
    - confidential
    - private

    private:
    - only owner_user_id can retrieve it.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF (.pdf)")

    try:
        access_metadata = build_document_metadata(
            doc_id=doc_id,
            visibility=visibility,
            owner_user_id=owner_user_id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        add_pdf_to_db(
            tmp_path,
            access_metadata=access_metadata,
        )

        return {
            "status": "success",
            "message": f"Đã lưu '{file.filename}' vào DB thành công.",
            "filename": file.filename,
            "metadata": access_metadata,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý PDF: {str(e)}")

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# API 2: Query RAG → trả lời như chatbot
# ─────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, summary="Hỏi RAG chatbot")
async def query_rag(request: QueryRequest):
    try:
        result = agent_graph.invoke(
            {
                "question": request.question,
                "messages": [HumanMessage(content=request.question)],
                "user_id": request.user_id,
                "role": request.role,
            },
            config={"configurable": {"thread_id": request.thread_id}},
        )

        print("Result keys:", list(result.keys()))

        # Lấy tin nhắn cuối cùng của AI từ messages
        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        if not answer:
            answer = (
                result.get("answer")
                or result.get("generation")
                or result.get("output")
                or result.get("response")
                or "Không có câu trả lời."
            )

        # Lấy sources nếu có
        sources = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                    if isinstance(payload, dict) and payload.get("sources"):
                        for src in payload["sources"]:
                            if src:
                                sources.append(src)
                except Exception:
                    pass

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi query RAG: {str(e)}")


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Chạy trực tiếp: python api.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )