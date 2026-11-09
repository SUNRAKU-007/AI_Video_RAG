import os
import tempfile
import shutil

from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from pydub import AudioSegment


# ==================================================
# CONFIGURATION
# ==================================================

DOWNLOAD_DIR = Path(
    os.getenv("AUDIO_WORK_DIR", "downloads")
)

DEFAULT_CHUNK_MINUTES = 10

SAMPLE_RATE = 16000

AUDIO_CHANNELS = 1

AUDIO_SAMPLE_WIDTH = 2

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# URL VALIDATION
# ==================================================

def is_url(source: str) -> bool:

    parsed = urlparse(source)

    return parsed.scheme in ("http", "https")


# ==================================================
# YOUTUBE AUDIO DOWNLOADER
# ==================================================

def download_youtube_audio(
    url: str,
    output_dir: str | Path | None = None
) -> str:

    output_dir = Path(
        output_dir or DOWNLOAD_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_template = str(
        output_dir / "%(id)s.%(ext)s"
    )

    ydl_opts = {

        "format": "bestaudio/best",

        "outtmpl": output_template,

        "noplaylist": True,

        "quiet": True,

        "no_warnings": True,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }
        ]

    }

    print("Downloading YouTube audio...")

    try:

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            video_id = info.get("id")

            if not video_id:

                raise RuntimeError(
                    "Could not identify the downloaded video."
                )

            wav_path = output_dir / f"{video_id}.wav"

            if wav_path.is_file():

                print(
                    "YouTube audio downloaded successfully."
                )

                return str(wav_path)

            candidates = list(
                output_dir.glob(f"{video_id}.*")
            )

            if len(candidates) == 1:

                return str(candidates[0])

            raise FileNotFoundError(
                "Downloaded audio file could not be located."
            )

    except yt_dlp.utils.DownloadError as e:

        raise RuntimeError(
            f"YouTube download failed: {e}"
        ) from e


# ==================================================
# WAV CONVERSION
# ==================================================

def convert_to_wav(
    input_path: str,
    output_dir: str | Path | None = None
) -> str:

    input_path = Path(input_path)

    if not input_path.is_file():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_dir = Path(
        output_dir or input_path.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / f"{input_path.stem}_converted.wav"
    )

    print("Converting audio to WAV...")

    try:

        audio = AudioSegment.from_file(
            str(input_path)
        )

        if len(audio) == 0:

            raise ValueError(
                "Input audio is empty."
            )

        audio = (
            audio
            .set_channels(AUDIO_CHANNELS)
            .set_frame_rate(SAMPLE_RATE)
            .set_sample_width(AUDIO_SAMPLE_WIDTH)
        )

        audio.export(
            str(output_path),
            format="wav"
        )

        print(
            f"WAV conversion completed: {output_path}"
        )

        return str(output_path)

    except Exception as e:

        output_path.unlink(missing_ok=True)

        raise RuntimeError(
            f"Audio conversion failed for {input_path}. "
            f"Check that FFmpeg is installed and the input is valid. "
            f"Details: {e}"
        ) from e


# ==================================================
# AUDIO CHUNKING
# ==================================================

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES
) -> list[str]:

    if chunk_minutes <= 0:

        raise ValueError(
            "chunk_minutes must be greater than zero."
        )

    wav_path = Path(wav_path)

    if not wav_path.is_file():

        raise FileNotFoundError(
            f"WAV file not found: {wav_path}"
        )

    print("Chunking audio...")

    try:

        audio = AudioSegment.from_wav(
            str(wav_path)
        )

    except Exception as e:

        raise RuntimeError(
            f"Unable to read WAV file: {e}"
        ) from e

    if len(audio) == 0:

        raise ValueError(
            "Cannot chunk an empty audio file."
        )

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    total_chunks = (
        len(audio) + chunk_ms - 1
    ) // chunk_ms

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            wav_path.parent / f"chunk_{i:04d}.wav"
        )

        chunk.export(
            str(chunk_path),
            format="wav"
        )

        chunks.append(str(chunk_path))

        print(
            f"Created chunk {i + 1}/{total_chunks}"
        )

    print(
        f"Audio ready: {len(chunks)} chunks created."
    )

    return chunks


# ==================================================
# MAIN INPUT PROCESSOR
# ==================================================

def process_input(
    source: str,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES
) -> list[str]:

    if not isinstance(source, str) or not source.strip():

        raise ValueError(
            "Please provide a valid URL or file path."
        )

    source = source.strip()

    # Create a unique workspace for this job.

    job_dir = Path(
        tempfile.mkdtemp(
            prefix="job_",
            dir=str(DOWNLOAD_DIR)
        )
    )

    downloaded_path = None

    print("\nStarting audio processing...")

    try:

        # ------------------------------------------
        # STEP 1: IDENTIFY INPUT
        # ------------------------------------------

        if is_url(source):

            print(
                "Detected YouTube URL. Downloading audio..."
            )

            downloaded_path = download_youtube_audio(
                source,
                output_dir=job_dir
            )

            input_path = Path(downloaded_path)

        else:

            print(
                "Detected local file."
            )

            input_path = Path(
                source
            ).expanduser()

            if not input_path.is_file():

                raise FileNotFoundError(
                    f"Local file does not exist: {input_path}"
                )

        # ------------------------------------------
        # STEP 2: CONVERT TO WAV
        # ------------------------------------------

        wav_path = convert_to_wav(
            str(input_path),
            output_dir=job_dir
        )

        # Delete only the file downloaded by yt-dlp.
        # NEVER delete the user's original local file.

        if downloaded_path:

            Path(downloaded_path).unlink(
                missing_ok=True
            )

        # ------------------------------------------
        # STEP 3: CHUNK AUDIO
        # ------------------------------------------

        chunks = chunk_audio(
            wav_path,
            chunk_minutes
        )

        if not chunks:

            raise RuntimeError(
                "Audio chunking did not generate any files."
            )

        # Full WAV is no longer required.

        Path(wav_path).unlink(
            missing_ok=True
        )

        print(
            f"\nAudio processing completed. "
            f"{len(chunks)} chunks ready."
        )

        return chunks

    except Exception:

        # Remove incomplete processing workspace.

        shutil.rmtree(
            job_dir,
            ignore_errors=True
        )

        raise


# ==================================================
# AUDIO CLEANUP
# ==================================================

def cleanup_processed_audio(
    chunks: list[str]
) -> None:

    if not chunks:

        return

    job_dir = Path(
        chunks[0]
    ).resolve().parent

    downloads_root = DOWNLOAD_DIR.resolve()

    # Safety validation.

    if job_dir.parent != downloads_root:

        raise ValueError(
            "Cleanup refused: directory outside DOWNLOAD_DIR."
        )

    if not job_dir.name.startswith("job_"):

        raise ValueError(
            "Cleanup refused: invalid job directory."
        )

    for chunk in chunks:

        if Path(chunk).resolve().parent != job_dir:

            raise ValueError(
                "Cleanup refused: chunks belong to different jobs."
            )

    print(
        f"Cleaning temporary audio files: {job_dir}"
    )

    shutil.rmtree(
        job_dir
    )

    print(
        "Audio cleanup completed."
    )