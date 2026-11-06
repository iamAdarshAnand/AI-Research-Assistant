# AI Research Assistant 📚

A local-first PDF research assistant using **semantic retrieval** and **page-level citations**. Upload PDFs, ask questions, and inspect the exact passages supporting the answer.

> This version provides grounded extractive answers, not generated prose. It does not require a paid LLM API or send documents to a hosted model.

## Features
- Multi-PDF upload with per-page text extraction
- Local sentence-transformer embeddings and cosine similarity
- Search across document chunks
- Answers built from retrieved source passages with file and page citations
- Downloadable cited research notes
- Streamlit UI and pytest tests

## Quick start
```bash
git clone https://github.com/iamAdarshAnand/AI-Research-Assistant.git
cd AI-Research-Assistant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The embedding model is downloaded on first use, so internet is needed once. Later runs can use the cached model. PDFs must contain selectable text; scanned/image-only PDFs require OCR, which is not included.

## Architecture
```
PDF uploads -> PyMuPDF page extraction -> overlapping text chunks
            -> sentence-transformer embeddings -> cosine similarity
Question -> query embedding -> ranked passages -> cited extractive answer
```

## Testing
```bash
pytest -q
```

## Limitations
- Retrieved passages may be irrelevant; verify citations before relying on them.
- No OCR, multimodal parsing, or generative synthesis in this version.
- Uploaded files are processed in the running app session; avoid sensitive documents on untrusted hosting.

Built by Adarsh Anand for educational and research use.
