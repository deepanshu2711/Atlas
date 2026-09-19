from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from sqlmodel import Session

from app.repositories.documents import DocumentsRepository
from app.schemas.query import QueryPayload
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
                 mode: RetrievalMode | None = None) -> list[Document]:
        return retrieve(payload.query, payload.document_id,
                        use_v2=payload.use_v2, k=k, mode=mode)

    async def query(self, payload: QueryPayload):
        document = self.document_repository.find_by_id(payload.document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        docs = self.retrieve(payload)

        if not docs:
            return {"answer": "I don't have enough information in this document to answer that.", "sources": []}

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        messages = _ANSWER_PROMPT.format_messages(
            context=context, question=payload.query)
        response = await llm.ainvoke(messages)

        return {"answer": response.content, "sources": docs}
