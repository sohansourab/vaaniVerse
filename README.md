# VāṇiVerse 🏛️
### Reimagining Cultural Knowledge Retrieval for India's Linguistic Diversity

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)](https://streamlit.io)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3-orange)](https://huggingface.co/BAAI/bge-m3)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

---

# VāṇiVerse

### Reimagining Cultural Knowledge Retrieval for India's Linguistic Diversity

VāṇiVerse is an open-source multilingual Retrieval-Augmented Generation (RAG) platform designed to preserve, explore, and interact with India's vast linguistic and cultural knowledge through Artificial Intelligence.

Modern AI systems perform exceptionally well on globally digitized content but struggle with regional languages, transliterated text, historical documents, manuscripts, and culturally specific knowledge sources. VāṇiVerse addresses this challenge by combining multilingual retrieval, document intelligence, semantic search, and culturally-aware AI reasoning into a unified knowledge platform.

The project begins with Telugu language support and is designed to scale across Indic languages and heritage datasets.

---

## Vision

India possesses centuries of knowledge distributed across:

* Regional literature
* Manuscripts and inscriptions
* Religious and philosophical texts
* Educational archives
* Community-generated knowledge repositories

Much of this information remains inaccessible to modern AI systems due to language diversity, script variations, and limited digitization.

VāṇiVerse aims to bridge this gap by creating an AI-powered cultural knowledge retrieval ecosystem capable of understanding, retrieving, and reasoning over multilingual heritage content.

---

## Key Challenges

| Challenge                | Example                                                                           |
| ------------------------ | --------------------------------------------------------------------------------- |
| Linguistic Fragmentation | `rama evaru` and `రామ ఎవరు` represent the same query in different scripts         |
| Script Variability       | Telugu, Romanized Telugu, and mixed-language inputs                               |
| Cultural Context Loss    | Heritage documents contain concepts poorly represented in modern datasets         |
| Preservation Gap         | Large collections remain available only as scanned documents or physical archives |

---

## Core Features

### Multilingual Retrieval

* Native Telugu support
* Romanized Telugu query understanding
* English query support
* Script-aware retrieval pipeline

### Cultural Knowledge Search

* Source-grounded answers
* Citation-backed responses
* Hallucination-resistant prompting
* Heritage-focused document retrieval

### Document Intelligence

* PDF ingestion and parsing
* Unicode normalization
* Script detection
* Semantic chunk generation

### Flexible AI Backends

* Groq-powered online inference
* Ollama-powered local inference
* Swappable embedding and vector storage layers

### Production-Ready API

* FastAPI backend
* RESTful endpoints
* Modular architecture
* Extensible ingestion pipeline

---

## System Architecture

```text
Document
   │
   ▼
Parser
   │
   ▼
Normalizer
   │
   ▼
Chunker
   │
   ▼
BGE-M3 Embeddings
   │
   ▼
FAISS / Qdrant
   │
   ▼
Retriever
   │
   ▼
Prompt Builder
   │
   ▼
LLM (Groq / Ollama)
   │
   ▼
Grounded Response + Citations
```

---

## Project Structure

```text
vaniverse/
├── api/
│   └── main.py
│
├── core/
│   ├── config.py
│   └── exceptions.py
│
├── ingestion/
│   ├── parser.py
│   ├── normalizer.py
│   └── chunker.py
│
├── retrieval/
│   ├── embeddings.py
│   └── vector_store.py
│
├── reasoning/
│   ├── prompter.py
│   └── generator.py
│
├── frontend/
│   └── app.py
│
└── scripts/
    └── ingest_documents.py
```

---

## Technology Stack

| Layer               | Technology                       |
| ------------------- | -------------------------------- |
| Language            | Python 3.12                      |
| API Framework       | FastAPI                          |
| Frontend            | Streamlit                        |
| Embeddings          | BAAI BGE-M3                      |
| Vector Database     | FAISS / Qdrant                   |
| Document Processing | PyMuPDF                          |
| NLP                 | spaCy + Custom Telugu Processing |
| Online LLM          | Groq                             |
| Offline LLM         | Ollama                           |

---

## Retrieval Pipeline

### Parsing

* PDF extraction using PyMuPDF
* Text and layout processing
* Image boundary detection

### Normalization

* Unicode normalization
* Script identification
* Telugu text cleanup
* Romanized Telugu conversion

### Chunking

* Telugu-aware sentence segmentation
* Semantic chunk construction
* Overlap-aware retrieval optimization

### Retrieval

* Multilingual embeddings via BGE-M3
* Dense vector similarity search
* Metadata-aware ranking

### Generation

* Source-grounded prompting
* Script-aware response generation
* Citation-enforced answers

---

## Example Query Modes

| Input Type       | Example                           |
| ---------------- | --------------------------------- |
| Telugu           | `నేను ఎవరు అంటే ఏమిటి`            |
| Romanized Telugu | `nenu evarini`                    |
| English          | `What is the nature of the self?` |

All query types pass through a unified multilingual retrieval workflow while preserving language-specific context.

---

## API Endpoints

| Method | Endpoint       | Purpose                   |
| ------ | -------------- | ------------------------- |
| GET    | `/health`      | Service health and status |
| POST   | `/ingest`      | Document ingestion        |
| POST   | `/query`       | End-to-end RAG query      |
| GET    | `/search`      | Retrieval-only search     |
| GET    | `/index/stats` | Index statistics          |
| DELETE | `/index/reset` | Reset vector index        |

---

## Roadmap

### Phase 1 — Foundation

* Multilingual RAG pipeline
* Telugu-aware retrieval
* FastAPI backend
* Streamlit interface
* FAISS and Qdrant support

### Phase 2 — Robustness

* OCR integration
* Improved transliteration
* Advanced PDF layout handling
* Enhanced token estimation

### Phase 3 — Cultural Intelligence

* Temple inscription understanding
* Historical document processing
* Heritage knowledge graph
* Community contribution workflows

### Phase 4 — Indic Expansion

* Hindi support
* Tamil support
* Kannada support
* Sanskrit support
* Cross-language retrieval

---

## Contributing

Contributions are welcome from developers, researchers, linguists, and cultural preservation enthusiasts.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

Areas of interest include:

* Indic NLP
* OCR systems
* Multilingual retrieval
* Knowledge graphs
* Historical document processing
* Cultural AI research

---

## License

Released under the MIT License.

---

## Acknowledgements

Built using:

* BAAI BGE-M3
* FastAPI
* Streamlit
* PyMuPDF
* Groq
* Ollama
* FAISS
* Qdrant

---

### Preserving India's knowledge systems through multilingual AI.
