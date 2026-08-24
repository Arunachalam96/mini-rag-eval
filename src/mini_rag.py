"""
mini_rag.py - a minimal, dependency-light RAG you can read in one sitting.
 
Answers questions about ONE PDF, with the retrieval math done by hand (numpy)
so nothing is hidden behind a vector-database library. This is the "see the
mechanics" companion to the FAISS-based P1 pipeline.
 
Setup:
    pip install pypdf sentence-transformers numpy ollama
    ollama pull llama3.1:8b          # or any local model you already have
 
Run:
    python mini_rag.py paper.pdf
"""
 
import sys
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama
 
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.1:8b"
CHUNK_WORDS = 200
OVERLAP = 40
TOP_K = 4
 
 
# ---------- 1. Load ----------
def load_pdf(path):
    reader = PdfReader(path)
    # extract_text() returns None on image-only pages -> guard with "or ''"
    return "\n".join(page.extract_text() or "" for page in reader.pages)
 
 
# ---------- 2. Chunk ----------
def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    chunks, i = [], 0
    step = chunk_words - overlap          # overlap keeps ideas that straddle a cut
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_words]))
        i += step
    return chunks
 
 
# ---------- 3. Embed ----------
def embed(model, texts):
    # normalize_embeddings=True -> every vector has length 1
    # -> a dot product between two of them IS their cosine similarity
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
 
 
# ---------- 4. Retrieve (the heart of RAG, by hand) ----------
def retrieve(query, model, chunks, chunk_vecs, k=TOP_K):
    q = embed(model, [query])[0]          # shape (dim,)
    scores = chunk_vecs @ q               # shape (n_chunks,): one cosine score per chunk
    top = np.argsort(scores)[::-1][:k]    # indices of the k highest scores
    return [(chunks[i], float(scores[i])) for i in top]
 
 
# ---------- 5. Generate (grounded) ----------
PROMPT = """You are a careful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "Not in the document."
 
Context:
{context}
 
Question: {question}
Answer:"""
 
 
def answer(query, hits):
    context = "\n\n---\n\n".join(text for text, _ in hits)
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=query)}],
        options={"temperature": 0.1},     # low temp -> less invention
    )
    return resp["message"]["content"]
 
 
# ---------- glue ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python mini_rag.py <file.pdf>")
        return
    pdf_path = sys.argv[1]
 
    print("Loading + chunking...")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunks")
 
    print("Embedding (first run downloads the model)...")
    model = SentenceTransformer(EMBED_MODEL)
    chunk_vecs = embed(model, chunks)
 
    print("Ready. Ask questions (Ctrl-C to quit).\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        hits = retrieve(q, model, chunks, chunk_vecs)
        # Uncomment to SEE what retrieval actually found -- your #1 debugging tool:
        # for text, score in hits:
        #     print(f"  [{score:.3f}] {text[:120]}...")
        print("\nA:", answer(q, hits), "\n")
 
 
if __name__ == "__main__":
    main()
 