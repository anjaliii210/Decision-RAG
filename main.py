import logging
import datetime
import uuid

from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAQQueryResult,
)

from agent import RAGAgent

load_dotenv()


# Inngest Client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# PDF INGESTION
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2,
        period=datetime.timedelta(minutes=1),
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load():
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id,
        )

    def _upsert(chunks_and_src):

        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vectors = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(
            ids,
            vectors,
            payloads,
        )

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    chunks = await ctx.step.run(
        "load-and-chunk",
        _load,
        output_type=RAGChunkAndSrc,
    )

    result = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks),
        output_type=RAGUpsertResult,
    )

    return result.model_dump()


# AGENTIC QUERY
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    question = ctx.event.data["question"]

    agent = RAGAgent(ctx)

    result = await agent.run(question)

    return RAQQueryResult(
        answer=result["answer"],
        sources=result["sources"],
        num_contexts=len(result["contexts"]),
    ).model_dump()

# FastAPI
app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    [
        rag_ingest_pdf,
        rag_query_pdf_ai,
    ],
)