import os

from concurrent.futures import ThreadPoolExecutor

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 6000
CHUNK_OVERLAP = 300
REDUCE_BATCH_SIZE = 4


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

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )

    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.3,
        thinking_level=os.getenv(
            "GEMINI_THINKING_LEVEL",
            "low"
        )
    )


# --------------------------------------------------
# SPLIT TRANSCRIPT
# --------------------------------------------------

def split_transcript(transcript: str) -> list:

    if not transcript or not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_text(transcript)


# --------------------------------------------------
# SUMMARY CHAIN
# --------------------------------------------------

def build_summary_chain():

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting and video summarizer.

            Create a clear, informative and professional summary
            of the provided transcript.

            Preserve:
            - Important information
            - Main topics
            - Key explanations
            - Important examples
            - Conclusions

            Avoid unnecessary repetition.
            Use readable headings and bullet points where appropriate.
            Do not invent information.
            """
        ),
        (
            "human",
            "{text}"
        )
    ])

    return prompt | get_llm() | StrOutputParser()


# --------------------------------------------------
# SUMMARIZE ONE CHUNK
# --------------------------------------------------

def summarize_chunk(chunk: str) -> str:

    chain = build_summary_chain()

    return chain.invoke({
        "text": chunk
    })


# --------------------------------------------------
# DIRECT SUMMARIZATION
# --------------------------------------------------

def direct_summarize(transcript: str) -> str:

    if not transcript or not transcript.strip():
        return "No transcript available."

    return summarize_chunk(transcript)


# --------------------------------------------------
# REDUCE PARTIAL SUMMARIES
# --------------------------------------------------

def reduce_summaries(summaries: list) -> str:

    if not summaries:
        return "No summary available."

    while len(summaries) > 1:

        reduced = []

        for i in range(
            0,
            len(summaries),
            REDUCE_BATCH_SIZE
        ):

            batch = summaries[
                i:i + REDUCE_BATCH_SIZE
            ]

            if len(batch) == 1:

                reduced.append(batch[0])
                continue

            combined = "\n\n".join(batch)

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    You are an expert summarization assistant.

                    Combine the provided partial summaries into
                    one coherent and professional summary.

                    Preserve all important information.
                    Remove repetition.
                    Do not invent facts.
                    Use headings and bullet points where useful.
                    """
                ),
                (
                    "human",
                    "{text}"
                )
            ])

            chain = (
                prompt
                | get_llm()
                | StrOutputParser()
            )

            result = chain.invoke({
                "text": combined
            })

            reduced.append(result)

        summaries = reduced

    return summaries[0]


# --------------------------------------------------
# COMPLETE TRANSCRIPT SUMMARIZATION
# --------------------------------------------------

def summarize_transcript(transcript: str) -> str:

    if not transcript or not transcript.strip():
        return "No transcript available."

    if len(transcript) <= CHUNK_SIZE:
        return direct_summarize(transcript)

    chunks = split_transcript(transcript)

    max_workers = max(
        1,
        int(os.getenv("SUMMARY_MAX_WORKERS", "1"))
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        summaries = list(
            executor.map(
                summarize_chunk,
                chunks
            )
        )

    return reduce_summaries(summaries)


# --------------------------------------------------
# COMPATIBILITY FUNCTION
# --------------------------------------------------

def summarize(transcript: str) -> str:

    return summarize_transcript(transcript)


# --------------------------------------------------
# GENERATE VIDEO TITLE
# --------------------------------------------------

def generate_title(transcript: str) -> str:

    if not transcript or not transcript.strip():
        return "Untitled Video"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Based on the transcript, generate a short,
            professional and meaningful video title.

            Maximum 8 words.

            Return only the title.
            Do not add quotation marks or explanations.
            """
        ),
        (
            "human",
            "{text}"
        )
    ])

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    )

    title = chain.invoke({
        "text": transcript[:2000]
    })

    return title.strip().strip('"')