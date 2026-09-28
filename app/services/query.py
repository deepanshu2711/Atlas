from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from sqlmodel import Session

from app.core.config import settings
from app.repositories.documents import DocumentsRepository
from app.schemas.query import QueryPayload
from app.services.agent import NO_ANSWER, Trace, answer_agentic
from app.services.retrieval import RetrievalMode, retrieve
from app.utils.llm_factory import llm

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document question-answering assistant. Answer the user's "
     "question using ONLY the information in the provided context below.\n\n"
     "Rules:\n"
     "- If the answer is not contained in the context, say \"I don't have "
     "enough information in this document to answer that.\" Do not use "
     "outside knowledge.\n"
     "- Be concise and directly answer the question first, then add "
     "supporting detail if needed.\n"
     "- If the context includes conflicting information, point out the "
     "conflict rather than picking one side silently.\n"
     "- Do not mention \"the context\" or \"the provided text\" in your "
     "answer - respond as if you simply know the document.\n\n"
     "Context:\n{context}"),
    ("human", "Question: {question}"),
])


class QueryService:
    def __init__(self, session: Session) -> None:
        self.document_repository = DocumentsRepository(session)

    def retrieve(self, payload: QueryPayload, k: int | None = None,
                 mode: RetrievalMode | None = None,
                 use_rerank: bool | None = None) -> list[Document]:
        return retrieve(payload.query, payload.document_id,
                        use_v2=payload.use_v2, k=k, mode=mode,
                        use_rerank=use_rerank)

    async def query(self, payload: QueryPayload):
        document = self.document_repository.find_by_id(payload.document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        agentic = settings.agentic_enabled if payload.agentic is None else payload.agentic
        if agentic:
            if not payload.use_v2:
                raise HTTPException(status_code=400, detail="Agentic retrieval requires use_v2")
            answer, docs, trace = await answer_agentic(payload.query, payload.document_id)
            return {"answer": answer, "sources": docs, "trace": trace}

        # Traced like the agentic path so the two can be compared on cost.
        trace = Trace(mode="single_pass", hops=1)
        docs = self.retrieve(payload)

        if not docs:
            return {"answer": NO_ANSWER, "sources": [], "trace": trace.finish()}

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        messages = _ANSWER_PROMPT.format_messages(
            context=context, question=payload.query)
        answer = await trace.call(llm, "answer", messages)

        return {"answer": answer, "sources": docs, "trace": trace.finish()}
