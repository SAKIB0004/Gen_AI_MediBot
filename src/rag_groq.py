from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import GROQ_MODEL

SYSTEM_PROMPT = """
You are a medical-book question answering assistant.
Use ONLY the provided context.
If the answer is not in the context, say "I don't know based on the provided book."
Do not provide medical diagnosis.
Use a maximum of three sentences and keep the answer concise.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def build_groq_rag_chain(retriever):
    llm = ChatGroq(
        model=GROQ_MODEL,          # e.g. llama-3.1-70b-versatile
        temperature=0.2,
        max_tokens=450
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}\n{d.page_content}"
            for d in docs
        )

    # ✅ FIX: use retriever.invoke(query)
    retrieve_docs = RunnableLambda(lambda q: retriever.invoke(q))

    build_inputs = RunnableLambda(lambda x: {
        "question": x["q"],
        "docs": x["docs"],
        "context": format_docs(x["docs"])
    })

    generate_answer = RunnableLambda(lambda x: {
        "answer": llm.invoke(
            PROMPT.invoke({"question": x["question"], "context": x["context"]})
        ),
        "docs": x["docs"]
    })

    rag_chain = (
        {
            "docs": retrieve_docs,
            "q": RunnablePassthrough()
        }
        | build_inputs
        | generate_answer
    )

    return rag_chain
