import os
import tempfile
import threading
import logging

import requests

from pathlib import Path

from faster_whisper import WhisperModel

from pydub import AudioSegment


# ==================================================
# CONFIGURATION
# ==================================================

# Sarvam API accepts short audio segments.
SARVAM_PIECE_SECONDS = 25

SARVAM_STT_TRANSLATE_URL = (
    "https://api.sarvam.ai/speech-to-text-translate"
)

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v2.5"
)

# Faster-Whisper configuration.

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small"
)

WHISPER_DEVICE = os.getenv(
    "WHISPER_DEVICE",
    "cpu"
)

WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8"
)

WHISPER_CPU_THREADS = int(
    os.getenv(
        "WHISPER_CPU_THREADS",
        str(min(os.cpu_count() or 4, 8))
    )
)

WHISPER_BEAM_SIZE = int(
    os.getenv(
        "WHISPER_BEAM_SIZE",
        "3"
    )
)

# Cached model instance.

_model = None

_model_lock = threading.Lock()

logger = logging.getLogger(__name__)


# ==================================================
# FASTER-WHISPER MODEL LOADING
# ==================================================

def load_model():

    """
    Load Faster-Whisper once and reuse the model.

    Default configuration:
        Model: small
        Device: CPU
        Compute type: INT8

    Model is cached to avoid repeated initialization.
    """

    global _model

    if _model is None:

        with _model_lock:

            # Double-check after acquiring the lock.

            if _model is None:

                logger.info(
                    "Loading Faster-Whisper model: %s",
                    WHISPER_MODEL
                )

                try:

                    _model = WhisperModel(

                        WHISPER_MODEL,

                        device=WHISPER_DEVICE,

                        compute_type=WHISPER_COMPUTE_TYPE,

                        cpu_threads=WHISPER_CPU_THREADS,

                        num_workers=1

                    )

                    logger.info(
                        "Faster-Whisper loaded successfully."
                    )

                    logger.info(
                        "Device: %s | Compute: %s | Threads: %s",
                        WHISPER_DEVICE,
                        WHISPER_COMPUTE_TYPE,
                        WHISPER_CPU_THREADS
                    )

                except Exception as e:

                    raise RuntimeError(
                        f"Unable to load Faster-Whisper: {e}"
                    ) from e

    return _model


# ==================================================
# WHISPER TRANSCRIPTION
# ==================================================

def transcribe_chunk_whisper(
    chunk_path: str,
    translate: bool = False,
    language: str = "en"
) -> str:

    """
    Transcribe one audio chunk using Faster-Whisper.

    Args:
        chunk_path: Audio file path.
        translate: Translate audio into English.
        language: Source language code.

    Returns:
        Transcribed text.
    """

    if not os.path.isfile(chunk_path):

        raise FileNotFoundError(
            f"Audio chunk not found: {chunk_path}"
        )

    model = load_model()

    task = (
        "translate"
        if translate
        else "transcribe"
    )

    logger.info(
        "Transcribing using Faster-Whisper: %s",
        chunk_path
    )

    try:

        segments, info = model.transcribe(

            chunk_path,

            language=None if translate else language,

            task=task,

            beam_size=WHISPER_BEAM_SIZE,

            vad_filter=True

        )

        # Important:
        # Faster-Whisper returns a generator.
        # Actual transcription happens while iterating.

        transcript_parts = []

        for segment in segments:

            text = segment.text.strip()

            if text:

                transcript_parts.append(text)

        transcript = " ".join(
            transcript_parts
        ).strip()

        logger.info(
            "Detected language: %s | Probability: %.2f",
            info.language,
            info.language_probability
        )

        return transcript

    except Exception as e:

        raise RuntimeError(
            f"Faster-Whisper transcription failed: {e}"
        ) from e


# ==================================================
# SARVAM API REQUEST
# ==================================================

def _send_to_sarvam(piece_path: str) -> str:

    """
    Send a short WAV segment to Sarvam API.
    Returns translated English transcript.
    """

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:

        raise RuntimeError(
            "SARVAM_API_KEY is missing from environment."
        )

    headers = {
        "api-subscription-key": api_key
    }

    data = {
        "model": SARVAM_MODEL,
        "with_diarization": "false"
    }

    try:

        with open(piece_path, "rb") as audio_file:

            files = {

                "file": (
                    os.path.basename(piece_path),
                    audio_file,
                    "audio/wav"
                )

            }

            response = requests.post(

                SARVAM_STT_TRANSLATE_URL,

                headers=headers,

                files=files,

                data=data,

                timeout=(15, 120)

            )

        if not response.ok:

            raise RuntimeError(
                f"Sarvam API returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:

            result = response.json()

        except ValueError as e:

            raise RuntimeError(
                "Sarvam returned invalid JSON."
            ) from e

        transcript = result.get("transcript")

        if not isinstance(transcript, str):

            raise RuntimeError(
                "Sarvam response does not contain a valid transcript."
            )

        return transcript.strip()

    except requests.Timeout as e:

        raise RuntimeError(
            "Sarvam API request timed out."
        ) from e

    except requests.RequestException as e:

        raise RuntimeError(
            f"Unable to connect to Sarvam API: {e}"
        ) from e


# ==================================================
# SARVAM TRANSCRIPTION
# ==================================================

def transcribe_chunk_sarvam(chunk_path: str) -> str:

    """
    Sarvam transcription for Hinglish audio.

    Splits audio into 25-second pieces,
    sends them to Sarvam, and combines results.
    """

    if not os.path.isfile(chunk_path):

        raise FileNotFoundError(
            f"Audio chunk not found: {chunk_path}"
        )

    if not os.getenv("SARVAM_API_KEY"):

        raise RuntimeError(
            "SARVAM_API_KEY is missing from environment."
        )

    try:

        audio = AudioSegment.from_wav(
            chunk_path
        )

    except Exception as e:

        raise RuntimeError(
            f"Unable to read audio file: {e}"
        ) from e

    if len(audio) == 0:

        raise ValueError(
            "Audio chunk is empty."
        )

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    transcripts = []

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[
            start:start + piece_ms
        ]

        piece_path = None

        try:

            # Unique temporary file.

            with tempfile.NamedTemporaryFile(

                prefix="sarvam_piece_",

                suffix=".wav",

                delete=False

            ) as temp_file:

                piece_path = temp_file.name

            piece.export(
                piece_path,
                format="wav"
            )

            logger.info(
                "Sarvam piece %d/%d",
                i + 1,
                total_pieces
            )

            text = _send_to_sarvam(
                piece_path
            )

            if text:

                transcripts.append(text)

        finally:

            # Remove temporary file.

            if (
                piece_path
                and os.path.exists(piece_path)
            ):

                os.remove(piece_path)

    return " ".join(
        transcripts
    ).strip()


# ==================================================
# TRANSCRIPTION ROUTER
# ==================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Route audio to the appropriate transcription engine.

    English:
        Faster-Whisper (local, INT8)

    Hinglish:
        Sarvam AI (API)
    """

    language = language.strip().lower()

    if language == "hinglish":

        return transcribe_chunk_sarvam(
            chunk_path
        )

    elif language == "english":

        return transcribe_chunk_whisper(
            chunk_path,
            language="en"
        )

    else:

        raise ValueError(
            f"Unsupported language: {language}"
        )


# ==================================================
# TRANSCRIBE ALL CHUNKS
# ==================================================

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    """
    Transcribe all audio chunks sequentially
    and combine the results.
    """

    if not chunks:

        raise ValueError(
            "No audio chunks provided."
        )

    language = language.strip().lower()

    if language not in ["english", "hinglish"]:

        raise ValueError(
            "Language must be english or hinglish."
        )

    engine = (
        "Sarvam AI"
        if language == "hinglish"
        else "Faster-Whisper"
    )

    print(
        f"\nUsing {engine} for transcription."
    )

    transcripts = []

    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):

        print(
            f"\nTranscribing chunk {i + 1}/{total_chunks}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        if text:

            transcripts.append(text)

        print(
            f"Chunk {i + 1}/{total_chunks} completed."
        )

    full_transcript = " ".join(
        transcripts
    ).strip()

    print(
        "\nTranscription completed successfully."
    )

    return full_transcript