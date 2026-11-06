import fitz
import numpy as np
import pytest
from retrieval import Passage, chunk_text, extract_pdf, rank_passages, format_answer

def test_chunks_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij", "ij"]

def test_empty_text():
    assert chunk_text("  \n  ") == []

def test_bad_chunk_settings():
    with pytest.raises(ValueError):
        chunk_text("text", 4, 4)

def test_pdf_extraction():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Research methods and results")
    data = document.tobytes()
    document.close()
    passages = extract_pdf(data, "example.pdf")
    assert passages
    assert passages[0].page == 1
    assert "Research methods" in passages[0].text

class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.asarray([[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts])

def test_ranked_results():
    passages = [Passage("a.pdf", 1, "banana"), Passage("b.pdf", 2, "apple")]
    ranked = rank_passages("apple", passages, FakeModel(), top_k=1)
    assert ranked[0][0].filename == "b.pdf"
    assert "b.pdf, page 2" in format_answer(ranked)
