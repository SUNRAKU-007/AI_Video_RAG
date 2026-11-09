import os
import json

from concurrent.futures import ThreadPoolExecutor

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500


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
# TRANSCRIPT CHUNKING
# --------------------------------------------------

def split_transcript(transcript: str) -> list:

    if not transcript or not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(transcript)


# --------------------------------------------------
# PARSE JSON RESPONSE
# --------------------------------------------------

def parse_json_response(response: str) -> dict:

    response = response.strip()

    if response.startswith("```"):

        response = response.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Gemini did not return a valid JSON object."
        )

    return json.loads(
        response[start:end + 1]
    )


# --------------------------------------------------
# NORMALIZE EXTRACTION RESULT
# --------------------------------------------------

def normalize_result(data: dict) -> dict:

    action_items = (
        data.get("action_items")
        or data.get("actionItems")
        or []
    )

    decisions = (
        data.get("key_decisions")
        or data.get("keyDecisions")
        or data.get("decisions")
        or []
    )

    questions = (
        data.get("open_questions")
        or data.get("openQuestions")
        or data.get("questions")
        or []
    )

    normalized_actions = []

    for item in action_items:

        if isinstance(item, dict):

            normalized_actions.append({
                "task": str(item.get("task", "Not specified")),
                "owner": str(item.get("owner", "Not specified")),
                "deadline": str(item.get("deadline", "Not specified"))
            })

        elif isinstance(item, str):

            normalized_actions.append({
                "task": item,
                "owner": "Not specified",
                "deadline": "Not specified"
            })

    return {
        "action_items": normalized_actions,
        "key_decisions": [
            str(item) for item in decisions
        ],
        "open_questions": [
            str(item) for item in questions
        ]
    }


# --------------------------------------------------
# BUILD EXTRACTION PROMPT
# --------------------------------------------------

def build_extraction_chain():

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting and video analyst.

            Analyze the provided transcript and extract:

            1. Action Items
            For each action item provide:
            - task
            - owner
            - deadline

            Use "Not specified" if the owner or deadline
            is not explicitly mentioned.

            2. Key Decisions
            Extract important decisions made in the discussion.

            3. Open Questions
            Extract unresolved questions and topics requiring
            further discussion.

            Return ONLY valid JSON using this exact structure:

            {{
                "action_items": [
                    {{
                        "task": "...",
                        "owner": "...",
                        "deadline": "..."
                    }}
                ],
                "key_decisions": [],
                "open_questions": []
            }}

            Do not invent information.
            Use empty arrays if nothing is found.
            """
        ),
        (
            "human",
            "{text}"
        )
    ])

    return (
        prompt
        | get_llm()
        | StrOutputParser()
    )


# --------------------------------------------------
# EXTRACT FROM ONE CHUNK
# --------------------------------------------------

def extract_chunk(chunk: str) -> dict:

    chain = build_extraction_chain()

    response = chain.invoke({
        "text": chunk
    })

    data = parse_json_response(response)

    return normalize_result(data)


# --------------------------------------------------
# MERGE MULTIPLE EXTRACTIONS
# --------------------------------------------------

def merge_extractions(results: list) -> dict:

    if not results:
        return {
            "action_items": [],
            "key_decisions": [],
            "open_questions": []
        }

    if len(results) == 1:
        return results[0]

    combined = json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting analyst.

            Combine the extracted information from different
            transcript sections into one final result.

            Remove duplicate action items, decisions and questions.

            Preserve:
            - Task descriptions
            - Owners
            - Deadlines
            - Important decisions
            - Unresolved questions

            Return ONLY valid JSON in this structure:

            {{
                "action_items": [
                    {{
                        "task": "...",
                        "owner": "...",
                        "deadline": "..."
                    }}
                ],
                "key_decisions": [],
                "open_questions": []
            }}

            Do not invent information.
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

    response = chain.invoke({
        "text": combined
    })

    return normalize_result(
        parse_json_response(response)
    )


# --------------------------------------------------
# COMPLETE INSIGHT EXTRACTION
# --------------------------------------------------

def extract_all_insights(transcript: str) -> dict:

    if not transcript or not transcript.strip():

        return {
            "action_items": [],
            "key_decisions": [],
            "open_questions": []
        }

    chunks = split_transcript(transcript)

    max_workers = max(
        1,
        int(os.getenv("EXTRACTION_MAX_WORKERS", "1"))
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        results = list(
            executor.map(
                extract_chunk,
                chunks
            )
        )

    return merge_extractions(results)


# --------------------------------------------------
# COMPATIBILITY WRAPPERS
# --------------------------------------------------

def extract_action_items(transcript: str) -> str:

    result = extract_all_insights(transcript)

    items = result["action_items"]

    if not items:
        return "No action items found."

    output = []

    for index, item in enumerate(items, 1):

        output.append(
            f"{index}. {item['task']}\n"
            f"   Owner: {item['owner']}\n"
            f"   Deadline: {item['deadline']}"
        )

    return "\n\n".join(output)


def extract_key_decisions(transcript: str) -> str:

    result = extract_all_insights(transcript)

    decisions = result["key_decisions"]

    if not decisions:
        return "No key decisions found."

    return "\n".join(
        f"{i}. {decision}"
        for i, decision in enumerate(decisions, 1)
    )


def extract_questions(transcript: str) -> str:

    result = extract_all_insights(transcript)

    questions = result["open_questions"]

    if not questions:
        return "No open questions found."

    return "\n".join(
        f"{i}. {question}"
        for i, question in enumerate(questions, 1)
    )