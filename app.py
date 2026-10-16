"""
Streamlit front-end for the document Q&A assistant.
Retrieves the most relevant chunks from the local index and asks
GPT to answer strictly from that context, citing the source file.
"""
import os
import numpy as np
import streamlit as st
from openai import OpenAI
from ingest import EMBED_MODEL, INDEX_PATH

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 4

client = OpenAI()


@st.cache_resource
def load_index():
    data = np.load(INDEX_PATH, allow_pickle=True)
    return data["vectors"], data["chunks"], data["sources"]


def embed_query(query: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def top_k_chunks(query_vec, vectors, chunks, sources, k=TOP_K):
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    scores = vectors @ query_vec / np.where(norms == 0, 1e-8, norms)
    idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], sources[i], float(scores[i])) for i in idx]


def answer_question(question: str, context_blocks):
    context = "\n\n".join(
        f"[{src}]\n{chunk}" for chunk, src, _ in context_blocks
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know. "
        "Cite the source file name in brackets.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


st.set_page_config(page_title="docqa-ai", page_icon="📄")
st.title("📄 docqa-ai — Document Q&A Assistant")
st.caption("Ask questions about the documents in sample_docs/. Answers are grounded in the source text, with citations.")

if not os.path.exists(INDEX_PATH):
    st.error("No index found. Run `python ingest.py` first to build the index from sample_docs/.")
    st.stop()

vectors, chunks, sources = load_index()

question = st.text_input("Ask a question about your documents:")
if question:
    with st.spinner("Searching documents..."):
        q_vec = embed_query(question)
        hits = top_k_chunks(q_vec, vectors, chunks, sources)
        answer = answer_question(question, hits)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources used"):
        for chunk, src, score in hits:
            st.markdown(f"**{src}** (relevance: {score:.2f})")
            st.caption(chunk[:300] + "...")
