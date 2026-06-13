---

## Tech Stack

| Layer | Technology |
|---|---|
| **Embedding Model** | [BAAI/BGE-M3](https://huggingface.co/BAAI/bge-m3) — 100+ languages, 1024-dim dense vectors |
| **Vector Store** | FAISS (local MVP) / Qdrant (production) |
| **Document Parsing** | PyMuPDF |
| **NLP** | spaCy + custom Telugu rule-based splitter |
| **LLM — Online** | Groq API (llama-3.3-70b-versatile) |
| **LLM — Offline** | Ollama (llama3, local) |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Language** | Python 3.12 |

---

## Quickstart

### Prerequisites
- Python 3.12+
- Git

### Installation

```bash
# Clone
git clone https://github.com/yourusername/vaaniVerse.git
cd vaaniVerse

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python3 -m spacy download en_core_web_sm
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here   # Get free at console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
LLM_BACKEND=groq                       # groq | ollama
VECTOR_BACKEND=faiss                   # faiss | qdrant
EMBEDDING_DEVICE=cpu                   # cpu | cuda
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=32
CHUNK_SIZE=256
CHUNK_OVERLAP=32
TOP_K=5
SCORE_THRESHOLD=0.45
API_HOST=0.0.0.0
API_PORT=8000
```

### Ingest a Document

```bash
# Single PDF
python3 -m scripts.ingest_documents --input data/documents/your_document.pdf

# Entire directory
python3 -m scripts.ingest_documents --input data/documents/

# Dry run (parse + chunk only, no embedding)
python3 -m scripts.ingest_documents --input data/documents/ --dry-run
```

### Start the API

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

### Launch the UI

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status + index statistics |
| `POST` | `/ingest` | Upload PDF → parse → chunk → embed → index |
| `POST` | `/query` | Full RAG: retrieve → prompt → LLM → answer + citations |
| `GET` | `/search` | Semantic retrieval only — no LLM, returns raw chunks |
| `GET` | `/index/stats` | Vector count + backend info |
| `DELETE` | `/index/reset` | Wipe in-memory index |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "నేను ఎవరు అంటే ఏమిటి",
    "top_k": 5,
    "score_threshold": 0.45
  }'
```

```json
{
  "answer": "నేను ఎవరు అనే ఆత్మ విచారణ మీ నిజమైన స్వభావాన్ని గ్రహించడానికి స్రళమైన మార్గం [SOURCE 1]...",
  "query_script": "telugu",
  "model": "llama-3.3-70b-versatile",
  "backend": "groq",
  "sources_used": [
    {
      "chunk_id": "document_p1_0",
      "source": "document.pdf",
      "page_number": 1,
      "score": 0.6311,
      "script": "telugu"
    }
  ],
  "total_chunks_retrieved": 3
}
```

---

## Multilingual Query Support

VāṇiVerse handles three query modes transparently:

| Mode | Example | Behavior |
|---|---|---|
| **Telugu script** | `నేను ఎవరు అంటే ఏమిటి` | Direct embedding, Telugu response |
| **Romanized Telugu** | `nenu evarini` | Phrase-map → Telugu script → embedding |
| **English** | `What is the nature of the self?` | Direct embedding, English response |

---

## Roadmap

### Phase 1 — Core RAG Pipeline ✅
- [x] PyMuPDF document parser
- [x] Telugu-aware chunker with danda sentence splitting
- [x] BGE-M3 multilingual embeddings
- [x] FAISS / Qdrant vector store
- [x] Source-grounded prompt templates
- [x] Groq / Ollama LLM backends
- [x] FastAPI REST layer
- [x] Streamlit UI
- [x] CLI batch ingestion

### Phase 2 — Multilingual Robustness 🔄
- [ ] OCR fallback via Tesseract (`tel` langpack) for scanned PDFs
- [ ] `indic-transliteration` library integration (replaces naive syllable map)
- [ ] BGE-M3 tokenizer for accurate token counting
- [ ] Multi-column PDF layout handling

### Phase 3 — Multimodal + Community
- [ ] Temple inscription image extraction pipeline
- [ ] Community document contribution handlers
- [ ] Heritage knowledge graph
- [ ] Historical timeline generation

---

## Contributing

VāṇiVerse is built for the community. Contributions welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit your changes
4. Open a Pull Request

Priority areas: Telugu NLP improvements, OCR pipeline, additional Indic language support.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built with [BAAI/BGE-M3](https://huggingface.co/BAAI/bge-m3),
[FastAPI](https://fastapi.tiangolo.com),
[Streamlit](https://streamlit.io),
[PyMuPDF](https://pymupdf.readthedocs.io),
and [Groq](https://console.groq.com).

---

*VāṇiVerse — preserving India's linguistic heritage, one document at a time.*