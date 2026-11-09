import os
import hashlib
import logging

from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "vector_db"))

COLLECTION_PREFIX = os.getenv(
    "COLLECTION_PREFIX",
    "meeting"
)

DEFAULT_COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    f"{COLLECTION_PREFIX}_default"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load the embedding model once and reuse it.

    Caching prevents repeatedly loading the model
    whenever a vector store is initialized.
    """

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": EMBEDDING_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# --------------------------------------------------
# TRANSCRIPT IDENTIFICATION
# --------------------------------------------------

def get_transcript_id(transcript: str) -> str:
    """
    Generate a stable identifier for a transcript.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    return hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()[:24]


def get_collection_name(transcript: str) -> str:
    """
    Generate a unique Chroma collection name for a transcript.
    """

    transcript_id = get_transcript_id(transcript)

    return f"{COLLECTION_PREFIX}_{transcript_id}"


# --------------------------------------------------
# TRANSCRIPT SPLITTING
# --------------------------------------------------

def split_transcript(transcript: str) -> list:
    """
    Split a transcript into smaller overlapping pieces.

    Overlap preserves context between adjacent chunks.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_text(transcript)


# --------------------------------------------------
# BUILD VECTOR STORE
# --------------------------------------------------

def build_vector_store(transcript: str):
    """
    Create or update a persistent Chroma vector store.

    Each transcript has its own collection.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Cannot build vector store from empty transcript.")

    transcript_id = get_transcript_id(transcript)
    collection_name = get_collection_name(transcript)

    logger.info(
        "Building vector store for transcript: %s",
        transcript_id
    )

    chunks = split_transcript(transcript)

    if not chunks:
        raise RuntimeError("No transcript chunks were generated.")

    documents = []

    for index, chunk in enumerate(chunks):

        document = Document(
            page_content=chunk,
            metadata={
                "transcript_id": transcript_id,
                "chunk_index": index,
                "source": "meeting_transcript"
            }
        )

        documents.append(document)

    ids = [
        f"{transcript_id}_{index:05d}"
        for index in range(len(documents))
    ]

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR)
    )

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    logger.info(
        "Vector store ready. Collection: %s | Chunks: %d",
        collection_name,
        len(documents)
    )

    return vector_store


# --------------------------------------------------
# LOAD EXISTING VECTOR STORE
# --------------------------------------------------

def load_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME
):
    """
    Load an existing Chroma collection.
    """

    if not collection_name:
        raise ValueError("A collection name must be provided.")

    logger.info(
        "Loading Chroma collection: %s",
        collection_name
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR)
    )


# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------

def get_retriever(
    vector_store,
    k: int = 4
):
    """
    Return a similarity-based retriever.

    k controls how many relevant chunks are retrieved
    for each user question.
    """

    if k < 1:
        raise ValueError("Retriever k must be at least 1.")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )