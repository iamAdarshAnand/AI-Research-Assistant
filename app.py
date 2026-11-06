"""Local-first, citation-oriented PDF research interface."""
import hashlib
import streamlit as st
from sentence_transformers import SentenceTransformer
from retrieval import extract_pdf, rank_passages, format_answer

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Research Assistant")
st.caption("Semantic PDF search with traceable page citations. No LLM API key required.")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

uploaded = st.file_uploader("Upload text-based PDF documents", type=["pdf"], accept_multiple_files=True)
question = st.text_input("Ask a question about the documents", placeholder="What does the report say about the main findings?")
top_k = st.slider("Passages to retrieve", 1, 10, 4)

if uploaded and question.strip():
    signature = hashlib.sha256()
    for file in uploaded:
        signature.update(file.name.encode("utf-8"))
        signature.update(file.getvalue())
    signature.update(question.encode("utf-8"))
    signature.update(str(top_k).encode("utf-8"))
    if st.button("🔎 Search documents", type="primary"):
        passages = []
        for file in uploaded:
            try:
                passages.extend(extract_pdf(file.getvalue(), file.name))
            except ValueError as exc:
                st.warning(str(exc))
        if not passages:
            st.warning("No searchable text found. Scanned PDFs may need OCR.")
        else:
            with st.spinner("Searching passages..."):
                results = rank_passages(question, passages, get_model(), top_k=top_k)
            answer = format_answer(results)
            st.subheader("Cited research notes")
            st.text(answer)
            st.download_button("Download notes", answer, file_name="research_notes.txt", mime="text/plain")
            st.caption("Similarity is a retrieval score, not a probability that a passage is correct.")
elif not uploaded:
    st.info("Upload at least one PDF to begin.")
