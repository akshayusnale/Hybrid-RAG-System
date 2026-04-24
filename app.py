# ===============================
# app.py — Hybrid RAG Streamlit UI
# ===============================

import time
import json
import re
import streamlit as st
import numpy as np

# -------------------------------
# Basic preprocessing (BM25)
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# -------------------------------
# Dense retrieval (FAISS)
# -------------------------------
def dense_retrieve(query, model, faiss_index, faiss_metadata, top_k=10):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "chunk_id": faiss_metadata[idx]["chunk_id"],
            "title": faiss_metadata[idx]["title"],
            "url": faiss_metadata[idx]["url"],
            "text": faiss_metadata[idx]["text"]
        })

    return results


# -------------------------------
# BM25 retrieval
# -------------------------------
def bm25_retrieve(query, chunks, bm25, top_k=10):
    query_tokens = preprocess_text(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        results.append({
            "rank": rank + 1,
            "score": float(scores[idx]),
            "chunk_id": chunks[idx]["chunk_id"],
            "title": chunks[idx]["title"],
            "url": chunks[idx]["url"],
            "text": chunks[idx]["text"]
        })

    return results


# -------------------------------
# Reciprocal Rank Fusion (RRF)
# -------------------------------
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    rrf_scores = {}

    for r in dense_results:
        rrf_scores[r["chunk_id"]] = rrf_scores.get(
            r["chunk_id"], 0
        ) + 1 / (k + r["rank"])

    for r in bm25_results:
        rrf_scores[r["chunk_id"]] = rrf_scores.get(
            r["chunk_id"], 0
        ) + 1 / (k + r["rank"])

    return sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


def build_chunk_lookup(chunks):
    return {c["chunk_id"]: c for c in chunks}


def deduplicate_by_url(results):
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            unique.append(r)
            seen.add(r["url"])
    return unique


def hybrid_retrieve(
    query,
    dense_model,
    faiss_index,
    faiss_metadata,
    bm25,
    chunks,
    top_k=10,
    top_n=8
):
    dense_results = dense_retrieve(
        query, dense_model, faiss_index, faiss_metadata, top_k
    )

    bm25_results = bm25_retrieve(
        query, chunks, bm25, top_k
    )

    fused = reciprocal_rank_fusion(dense_results, bm25_results)

    dense_rank_map = {r["chunk_id"]: r["rank"] for r in dense_results}
    bm25_rank_map = {r["chunk_id"]: r["rank"] for r in bm25_results}

    lookup = build_chunk_lookup(chunks)

    final = []
    for chunk_id, score in fused[:top_n]:
        c = lookup[chunk_id]
        final.append({
            "chunk_id": chunk_id,
            "title": c["title"],
            "url": c["url"],
            "text": c["text"],
            "rrf_score": score,
            "dense_rank": dense_rank_map.get(chunk_id),
            "bm25_rank": bm25_rank_map.get(chunk_id),
        })

    return deduplicate_by_url(final)


# -------------------------------
# LLM Answer Generation (Flan-T5)
# -------------------------------
def build_prompt(question, chunks):
    prompt = (
        "Answer the question using ONLY the context below. "
        "Do not add external knowledge. "
        "If the answer cannot be derived, say \"I don't know\".\n\n"
        "Context:\n"
    )
    for i, c in enumerate(chunks):
        prompt += f"[{i+1}] {c['text']}\n\n"

    prompt += f"Question: {question}\nAnswer:"
    return prompt


def generate_answer(question, chunks, tokenizer, llm):
    prompt = build_prompt(question, chunks)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=4,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# Load resources (CACHED)
# -------------------------------
@st.cache_resource
def load_resources():
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    index = faiss.read_index("indexes/faiss.index")

    with open("indexes/faiss_metadata.json") as f:
        faiss_metadata = json.load(f)

    with open("data/chunks.json") as f:
        chunks = json.load(f)

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")

    tokenized = [preprocess_text(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return dense_model, index, faiss_metadata, bm25, chunks, tokenizer, llm


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="Hybrid RAG QA System",
    layout="wide"
)

st.title("🔍 Hybrid RAG Question Answering System")
st.write("Dense Retrieval + BM25 + RRF + LLM (Flan-T5)")

dense_model, index, faiss_metadata, bm25, chunks, tokenizer, llm = load_resources()

query = st.text_input(
    "Enter your question:",
    placeholder="How does the transformer architecture handle long-range dependencies?"
)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        start = time.time()

        sources = hybrid_retrieve(
            query,
            dense_model,
            index,
            faiss_metadata,
            bm25,
            chunks,
            top_k=10,
            top_n=8
        )

        answer = generate_answer(query, sources, tokenizer, llm)

        elapsed = time.time() - start

        st.subheader("🧠 Answer")
        st.write(answer)
        st.caption(f"⏱ Response Time: {elapsed:.2f} seconds")

        st.subheader("📚 Retrieved Context")
        for i, s in enumerate(sources, start=1):
            with st.expander(f"Chunk {i}: {s['title']}"):
                st.markdown(
                    f"""
                    **URL:** {s['url']}  
                    **RRF Score:** {s['rrf_score']:.4f}  
                    **Dense Rank:** {s['dense_rank']}  
                    **BM25 Rank:** {s['bm25_rank']}
                    """
                )
                st.write(s["text"][:900] + "…")
