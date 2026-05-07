import os
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1"
)

EMBED_MODEL = "mistral-embed"
EMBED_DIM = 1024
BATCH_SIZE = 50

splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    texts = [d.text for d in docs if getattr(d, "text", None)]
    if not texts:
        raise ValueError("No text extracted from PDF")

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    if not chunks:
        raise ValueError("Chunking failed: no chunks created")

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("No texts provided for embedding")

    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        if not response.data:
            raise ValueError("Embedding API returned empty response")

        batch_embeddings = [item.embedding for item in response.data]

        # validation
        for emb in batch_embeddings:
            if len(emb) != EMBED_DIM:
                raise ValueError(f"Embedding dimension mismatch: got {len(emb)}, expected {EMBED_DIM}")

        all_embeddings.extend(batch_embeddings)

    if len(all_embeddings) != len(texts):
        raise ValueError("Mismatch between input texts and embeddings")

    return all_embeddings