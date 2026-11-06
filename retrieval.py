"""PDF extraction, overlapping chunking and semantic retrieval."""
from dataclasses import dataclass
import numpy as np
import fitz

@dataclass(frozen=True)
class Passage:
    filename: str
    page: int
    text: str

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Expected chunk_size > overlap >= 0")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    step = chunk_size - overlap
    return [cleaned[i:i + chunk_size] for i in range(0, len(cleaned), step) if cleaned[i:i + chunk_size].strip()]

def extract_pdf(data: bytes, filename: str) -> list[Passage]:
    if not data:
        raise ValueError("PDF is empty")
    passages = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_number, page in enumerate(doc, start=1):
                for text in chunk_text(page.get_text("text")):
                    passages.append(Passage(filename=filename, page=page_number, text=text))
    except Exception as exc:
        raise ValueError(f"Could not read PDF {filename}: {exc}") from exc
    return passages

def rank_passages(query: str, passages: list[Passage], model, top_k: int = 4):
    """Return (passage, cosine similarity) sorted by descending score."""
    if not query.strip() or not passages or top_k <= 0:
        return []
    texts = [p.text for p in passages]
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True))
    query_vector = np.asarray(model.encode([query], normalize_embeddings=True))[0]
    scores = vectors @ query_vector
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [(passages[int(i)], float(scores[int(i)])) for i in indices]

def format_answer(results: list[tuple[Passage, float]]) -> str:
    if not results:
        return "No searchable passages found. Try another question or a text-based PDF."
    lines = ["Relevant excerpts (verify each source before use):"]
    for index, (passage, score) in enumerate(results, start=1):
        lines.append(f"\n[{index}] {passage.filename}, page {passage.page} (similarity {score:.2f})\n{passage.text}")
    return "\n".join(lines)
