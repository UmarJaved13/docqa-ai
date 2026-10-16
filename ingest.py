"""
Splits PDFs/text files into chunks, embeds them with OpenAI, and
saves the chunks + vectors to a local .npz index for retrieval.
"""
import os
import json
import glob
import numpy as np
from pypdf import PdfReader
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
INDEX_PATH = "index.npz"

client = OpenAI()


def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_batch(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def build_index(doc_dir: str):
    files = glob.glob(os.path.join(doc_dir, "*"))
    all_chunks, all_sources = [], []

    for path in files:
        text = read_text(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(path)] * len(chunks))

    if not all_chunks:
        raise ValueError(f"No readable documents found in {doc_dir}")

    vectors = []
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        vectors.extend(embed_batch(all_chunks[i:i + batch_size]))

    np.savez(
        INDEX_PATH,
        vectors=np.array(vectors, dtype=np.float32),
        chunks=np.array(all_chunks, dtype=object),
        sources=np.array(all_sources, dtype=object),
    )
    print(f"Indexed {len(all_chunks)} chunks from {len(files)} files -> {INDEX_PATH}")


if __name__ == "__main__":
    build_index("sample_docs")
