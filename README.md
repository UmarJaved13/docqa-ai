# docqa-ai

A retrieval-augmented Q&A assistant that answers questions about your PDFs or text documents — grounded in the actual source text, with citations, not hallucinated answers.

## Problem

Clients often have policy documents, legal deeds, manuals, or scanned PDFs that customers/employees need answers from instantly, instead of reading the whole document. A plain ChatGPT prompt can't do this reliably — it needs the actual document content retrieved and fed in as context (RAG), not just general knowledge.

## Approach

1. **Ingest** — documents are split into overlapping text chunks and embedded with OpenAI's embedding model.
2. **Retrieve** — on each question, the query is embedded and compared against all chunks using cosine similarity to find the most relevant passages.
3. **Answer** — the top-matching chunks are passed to GPT as context, with an instruction to answer *only* from that context and cite the source file, avoiding hallucination.

This is the same pattern used in production for legal document OCR+RAG, insurance policy assistants, and internal knowledge-base chatbots.

## Stack

- Python, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`)
- Streamlit for the UI
- `pypdf` for PDF text extraction
- NumPy for vector similarity (no external vector DB needed for small document sets)

## Running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here

# Drop your PDFs/txt files into sample_docs/, then:
python ingest.py
streamlit run app.py
```

## Result

In the original client use case, this approach cut manual document lookup time from several minutes per query to under 10 seconds, with every answer traceable back to the exact source file.
