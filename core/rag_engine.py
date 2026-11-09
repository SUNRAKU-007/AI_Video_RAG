import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from core.vector_store import (
    build_vector_store,
    load_vector_store
)


# --------------------------------------------------
# GEMINI MODEL
# --------------------------------------------------

def get_llm():

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        ),
        api_key=api_key,
        temperature=0.3,
        thinking_level=os.getenv(
            "GEMINI_THINKING_LEVEL",
            "low"
        )
    )


# --------------------------------------------------
# FORMAT RETRIEVED DOCUMENTS
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# FORMAT CHAT HISTORY
# --------------------------------------------------

def format_chat_history(
    chat_history,
    current_question=None
):

    if not chat_history:
        return ""

    if isinstance(chat_history, str):
        return chat_history

    formatted = []

    for message in chat_history:

        if isinstance(message, dict):

            role = message.get("role", "user")
            content = message.get("content", "")

        else:

            role = getattr(
                message,
                "type",
                "user"
            )

            content = getattr(
                message,
                "content",
                ""
            )

        if isinstance(content, list):

            content = " ".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(part)
                for part in content
            )

        content = str(content).strip()

        if not content:
            continue

        if (
            current_question
            and role in ["user", "human"]
            and content == current_question
        ):
            continue

        formatted.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted[-8:])


# --------------------------------------------------
# REWRITE FOLLOW-UP QUESTIONS
# --------------------------------------------------

def rewrite_question(
    question: str,
    chat_history: str,
    llm=None
) -> str:

    if not chat_history:
        return question

    llm = llm or get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a question rewriting assistant.

            Given the conversation history and the user's latest
            question, rewrite the latest question as a standalone
            question that can be understood without previous messages.

            Do not answer the question.
            Return only the rewritten question.
            """
        ),
        (
            "human",
            """
            Conversation history:
            {history}

            Latest question:
            {question}
            """
        )
    ])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "history": chat_history,
        "question": question
    }).strip()


# --------------------------------------------------
# CREATE RAG CHAIN
# --------------------------------------------------

def create_rag_chain(
    retriever,
    llm
):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert AI video and meeting assistant.

            Answer the user's question using ONLY the retrieved
            transcript context.

            Conversation history is provided only to understand
            follow-up questions. It is not independent evidence.

            If the answer is not available in the transcript,
            clearly say:

            "I could not find this information in the video transcript."

            Be precise, helpful and concise.

            Do not invent facts.

            Retrieved transcript context:
            {context}

            Conversation history:
            {chat_history}
            """
        ),
        (
            "human",
            "{question}"
        )
    ])

    def retrieve_context(inputs):

        query = inputs.get(
            "retrieval_query",
            inputs["question"]
        )

        docs = retriever.invoke(query)

        return format_docs(docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(retrieve_context)
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# --------------------------------------------------
# BUILD RAG CHAIN FOR A TRANSCRIPT
# --------------------------------------------------

def build_rag_chain(
    transcript: str,
    k: int = 4
):

    if not transcript or not transcript.strip():
        raise ValueError(
            "Cannot build RAG pipeline without a transcript."
        )

    vector_store = build_vector_store(
        transcript
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    llm = get_llm()

    return create_rag_chain(
        retriever,
        llm
    )


# --------------------------------------------------
# LOAD EXISTING RAG CHAIN
# --------------------------------------------------

def load_rag_chain(
    transcript=None,
    collection_name=None,
    k=4
):

    if transcript:

        vector_store = build_vector_store(
            transcript
        )

    else:

        vector_store = load_vector_store(
            collection_name=collection_name
        )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    llm = get_llm()

    return create_rag_chain(
        retriever,
        llm
    )


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

def ask_question(
    rag_chain,
    question: str,
    chat_history=None
) -> str:

    question = question.strip()

    if not question:
        return "Please enter a question."

    history = format_chat_history(
        chat_history,
        current_question=question
    )

    llm = get_llm()

    if history:

        retrieval_query = rewrite_question(
            question,
            history,
            llm
        )

    else:

        retrieval_query = question

    answer = rag_chain.invoke({
        "question": question,
        "retrieval_query": retrieval_query,
        "chat_history": history
    })

    return answer