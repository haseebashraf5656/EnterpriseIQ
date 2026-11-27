"""EnterpriseIQ document Q&A UI."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
MAX_QUESTION_CHARS = 2000  # Settings.max_question_chars
CHAT_TIMEOUT_SECONDS = 120
RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "hybrid_reranked")


class BackendError(Exception):
    """User-facing API failure. Message is safe to display."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _url(path: str) -> str:
    """Build a backend URL from the configured base."""
    return f"{BACKEND_URL}{path}"


def _error_detail(response: requests.Response) -> str:
    """Text from a 400 or 422 `detail` field."""
    try:
        body = response.json()
    except ValueError:
        return "The request was rejected."
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                msg = item.get("msg") or item.get("message")
                parts.append(str(msg) if msg else str(item))
            else:
                parts.append(str(item))
        text = "; ".join(part for part in parts if part)
        if text:
            return text
    return "The request was rejected."


def get_health() -> dict[str, Any] | None:
    """GET /health. None when the backend cannot be reached."""
    try:
        response = requests.get(_url("/health"), timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    return data if isinstance(data, dict) else None


def upload_files(uploads: list[Any]) -> list[dict[str, Any]]:
    """POST /documents/upload as multipart field `files`."""
    payload = [
        ("files", (item.name, item.getvalue(), item.type or "application/octet-stream"))
        for item in uploads
    ]
    try:
        response = requests.post(_url("/documents/upload"), files=payload, timeout=120)
    except requests.Timeout as exc:
        raise BackendError("The upload timed out. Try again.") from exc
    except requests.RequestException as exc:
        raise BackendError("Could not reach the backend.") from exc
    if response.status_code in (400, 422):
        raise BackendError(_error_detail(response))
    if response.status_code >= 400:
        raise BackendError("The upload failed. Try again.")
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError("The upload failed. Try again.") from exc
    if not isinstance(data, list):
        raise BackendError("The upload failed. Try again.")
    return [item for item in data if isinstance(item, dict)]


def list_documents() -> list[dict[str, Any]]:
    """GET /documents."""
    try:
        response = requests.get(_url("/documents"), timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise BackendError("Could not load documents.") from exc
    if not isinstance(data, list):
        raise BackendError("Could not load documents.")
    return [item for item in data if isinstance(item, dict)]


def delete_document(document_id: str) -> None:
    """DELETE /documents/{id}."""
    try:
        response = requests.delete(_url(f"/documents/{document_id}"), timeout=30)
    except requests.Timeout as exc:
        raise BackendError("The request timed out. Try again.") from exc
    except requests.RequestException as exc:
        raise BackendError("Could not reach the backend.") from exc
    if response.status_code in (400, 422):
        raise BackendError(_error_detail(response))
    if response.status_code == 404:
        raise BackendError(_error_detail(response))
    if response.status_code >= 400:
        raise BackendError("Could not delete the document.")


def post_chat(question: str, mode: str) -> dict[str, Any]:
    """POST /chat. Raises BackendError with a displayable message."""
    try:
        response = requests.post(
            _url("/chat"),
            json={"question": question, "mode": mode},
            timeout=CHAT_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise BackendError("The request timed out. Try again.") from exc
    except requests.ConnectionError as exc:
        raise BackendError("Could not reach the backend. Try again.") from exc
    except requests.RequestException as exc:
        raise BackendError("Could not reach the backend. Try again.") from exc
    if response.status_code in (502, 503):
        raise BackendError("The answer service is temporarily unavailable, try again")
    if response.status_code in (400, 422):
        raise BackendError(_error_detail(response))
    if response.status_code >= 400:
        raise BackendError("The request failed. Try again.")
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError("The request failed. Try again.") from exc
    if not isinstance(data, dict) or "answer" not in data:
        raise BackendError("The request failed. Try again.")
    return data


def passages_for_citation(citation: dict[str, Any], sources: list[dict[str, Any]]) -> list[str]:
    """Source texts that share filename and page with a citation."""
    texts: list[str] = []
    for source in sources:
        if source.get("filename") == citation.get("filename") and source.get("page") == citation.get("page"):
            text = source.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def render_upload_results(items: list[dict[str, Any]]) -> None:
    """One line per uploaded file."""
    for item in items:
        name = str(item.get("filename") or "upload")
        error = item.get("error")
        if error:
            st.error(f"{name}: {error}")
        else:
            st.success(f"{name}: accepted")


def render_sources(citations: list[Any], sources: list[Any]) -> None:
    """Citations with passage text matched by filename and page."""
    st.subheader("Sources")
    source_rows = [row for row in sources if isinstance(row, dict)]
    shown = [row for row in citations if isinstance(row, dict)]
    if not shown:
        st.caption("No citations returned.")
        return
    for citation in shown:
        filename = citation.get("filename") or "document"
        page = citation.get("page")
        with st.expander(f"{filename} — page {page}"):
            for text in passages_for_citation(citation, source_rows):
                st.write(text)


def render_retrieval(retrieval: dict[str, Any]) -> None:
    """Method, top_k, and latency_ms as returned by the API."""
    with st.expander("Retrieval details"):
        st.write("method:", retrieval.get("method"))
        st.write("top_k:", retrieval.get("top_k"))
        latency = retrieval.get("latency_ms")
        st.write("latency_ms:")
        if isinstance(latency, dict):
            for stage, ms in latency.items():
                st.write(f"{stage}: {ms}")


def render_answer(data: dict[str, Any]) -> None:
    """Answer, evidence warning or sources, and retrieval metadata."""
    st.markdown(str(data.get("answer", "")))
    if data.get("insufficient_evidence"):
        st.warning("Not enough information in the documents to answer this.")
    else:
        citations = data.get("citations") if isinstance(data.get("citations"), list) else []
        sources = data.get("sources") if isinstance(data.get("sources"), list) else []
        render_sources(citations, sources)
    retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), dict) else {}
    render_retrieval(retrieval)


def render_message(message: dict[str, Any]) -> None:
    """One stored chat turn."""
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message.get("content") or "")
            return
        if message.get("error"):
            st.error(message["error"])
            return
        data = message.get("data")
        if isinstance(data, dict):
            render_answer(data)


def render_health() -> None:
    """Connected or not-reachable indicator."""
    if st.button("Refresh", key="health_refresh"):
        st.rerun()
    health = get_health()
    if health is None:
        st.error("Not reachable")
        st.caption("Backend not ready. It can take a few minutes to start")
        return
    st.success("Connected")


def render_upload() -> None:
    """Multi-file upload. One rejected file does not hide the others."""
    uploads = st.file_uploader(
        "Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Upload", disabled=not uploads):
        try:
            st.session_state.upload_results = upload_files(list(uploads or []))
        except BackendError as exc:
            st.session_state.upload_results = [{"filename": "upload", "error": exc.message}]
    results = st.session_state.get("upload_results") or []
    render_upload_results(results)


@st.fragment(run_every="3s")
def document_library() -> None:
    """Document list. Polls while any document is still processing."""
    if st.button("Refresh", key="docs_refresh"):
        st.rerun(scope="fragment")
    try:
        documents = list_documents()
    except BackendError as exc:
        st.error(exc.message)
        return
    if not documents:
        st.caption("No documents yet.")
        return
    for doc in documents:
        doc_id = str(doc.get("id") or "")
        filename = str(doc.get("filename") or "document")
        status = str(doc.get("status") or "")
        label = f"{filename} — {status}"
        if status == "failed" and doc.get("error_message"):
            label = f"{label}: {doc['error_message']}"
        left, right = st.columns([4, 1])
        left.write(label)
        if doc_id and right.button("Delete", key=f"delete-{doc_id}"):
            try:
                delete_document(doc_id)
            except BackendError as exc:
                st.error(exc.message)
            else:
                st.rerun(scope="fragment")


def ask(question: str, mode: str) -> None:
    """Validate length, call /chat, and append the turn."""
    text = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not text:
            message = "Enter a question."
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "error": message})
            return
        if len(text) > MAX_QUESTION_CHARS:
            message = f"Question exceeds {MAX_QUESTION_CHARS} characters."
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "error": message})
            return
        with st.spinner("Answering..."):
            try:
                data = post_chat(text, mode)
            except BackendError as exc:
                st.error(exc.message)
                st.session_state.messages.append({"role": "assistant", "error": exc.message})
                return
        render_answer(data)
        st.session_state.messages.append({"role": "assistant", "data": data})


def main() -> None:
    """Sidebar library and independent question chat."""
    st.set_page_config(page_title="EnterpriseIQ", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("upload_results", [])

    with st.sidebar:
        st.header("Backend")
        render_health()
        st.header("Upload")
        render_upload()
        st.header("Documents")
        document_library()
        st.header("Retrieval")
        mode = st.selectbox("Retrieval mode", RETRIEVAL_MODES, index=3)

    st.title("EnterpriseIQ")
    st.caption("Each question is answered independently (the backend has no memory).")
    for message in st.session_state.messages:
        render_message(message)
    question = st.chat_input("Ask a question about your documents")
    if question:
        ask(question, mode)


if __name__ == "__main__":
    main()
