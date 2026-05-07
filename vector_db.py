from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sqlalchemy import text


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=1024):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

        def search(self, query_vector, top_k: int = 5):
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k
            )

            if not results:
                raise ValueError("No results found in vector search")

            contexts = []
            sources = set()

            for r in results:
                print("SCORE:", r.score)

                payload = getattr(r, "payload", {}) or {}
                text = payload.get("text", "")
                source = payload.get("source", "")

                if not text:
                    print("WARNING: Missing text in payload")
                    continue

                contexts.append(text)
                sources.add(source)

            return {"contexts": contexts, "sources": list(sources)}