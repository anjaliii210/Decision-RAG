import os

from data_loader import embed_texts
from vector_db import QdrantStorage
from inngest.experimental import ai


class RAGAgent:

    def __init__(self, ctx):

        self.ctx = ctx
        self.store = QdrantStorage()

        self.adapter = ai.openai.Adapter(
            auth_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-small-latest",
            base_url="https://api.mistral.ai/v1",
        )

    ################################################

    async def planner(self, question):

        res = await self.ctx.step.ai.infer(
            "planner",
            adapter=self.adapter,
            body={
                "temperature":0,
                "messages":[
                    {
                        "role":"system",
                        "content":"Return only SEARCH or ANSWER."
                    },
                    {
                        "role":"user",
                        "content":question
                    }
                ]
            }
        )

        return res["choices"][0]["message"]["content"].strip()

    ################################################

    def retrieve(self, query):

        vec = embed_texts([query])[0]

        return self.store.search(vec)

    ################################################

    async def evaluate(self, question, contexts):

        context = "\n".join(contexts)

        res = await self.ctx.step.ai.infer(
            "evaluate",
            adapter=self.adapter,
            body={
                "temperature":0,
                "messages":[
                    {
                        "role":"system",
                        "content":"Reply ONLY YES or NO. Is the context sufficient?"
                    },
                    {
                        "role":"user",
                        "content":f"""
Question:

{question}

Context:

{context}
"""
                    }
                ]
            }
        )

        return res["choices"][0]["message"]["content"].strip()

    ################################################

    async def rewrite(self, question):

        res = await self.ctx.step.ai.infer(
            "rewrite-query",
            adapter=self.adapter,
            body={
                "temperature":0,
                "messages":[
                    {
                        "role":"system",
                        "content":"Rewrite for semantic retrieval only."
                    },
                    {
                        "role":"user",
                        "content":question
                    }
                ]
            }
        )

        return res["choices"][0]["message"]["content"].strip()

    ################################################

    async def answer(self, question, contexts):

        context = "\n\n".join(contexts)

        res = await self.ctx.step.ai.infer(
            "generate-answer",
            adapter=self.adapter,
            body={
                "temperature":0.2,
                "messages":[
                    {
                        "role":"system",
                        "content":"Answer ONLY using supplied context."
                    },
                    {
                        "role":"user",
                        "content":f"""
Context:

{context}

Question:

{question}
"""
                    }
                ]
            }
        )

        return res["choices"][0]["message"]["content"].strip()

    ################################################

    async def run(self, question):

        decision = await self.planner(question)

        if decision == "ANSWER":

            return {
                "answer":"No retrieval required.",
                "sources":[],
                "contexts":[]
            }

        retrieved = self.retrieve(question)

        sufficient = await self.evaluate(
            question,
            retrieved["contexts"]
        )

        if sufficient == "NO":

            better_query = await self.rewrite(question)

            retrieved = self.retrieve(better_query)

        answer = await self.answer(
            question,
            retrieved["contexts"]
        )

        return {
            "answer":answer,
            "sources":retrieved["sources"],
            "contexts":retrieved["contexts"]
        }