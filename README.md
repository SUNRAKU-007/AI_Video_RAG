# AI Video Assistant

An AI-powered video assistant that turns long-form videos into searchable, conversational knowledge. Provide a YouTube URL or a local video, generate a transcript, get a structured summary and key insights, then ask questions about the video using retrieval-augmented generation (RAG).

## Features

- **Multiple input options:** YouTube URL, uploaded video, or local video path.
- **Audio preprocessing:** Uses FFmpeg to normalize audio to mono, 16 kHz WAV and split long recordings into 10-minute chunks.
- **Speech-to-text:** Faster-Whisper for English transcription and Sarvam AI for Hinglish transcription.
- **AI summaries:** Gemini generates a title and a structured summary.
- **Insight extraction:** Identifies action items, key decisions, and open questions.
- **Conversational Q&A:** Ask questions grounded in the processed video's transcript through a RAG pipeline.
- **Persistent retrieval:** ChromaDB stores transcript chunks, with a separate collection for each transcript.
- **Streamlit interface:** Process videos, inspect results, chat with the transcript, and download generated outputs.
- **Docker support:** Run the application in a container with Docker Compose.

## Architecture

```text
YouTube URL / Uploaded Video / Local Video
                  |
                  v
          Audio Preprocessing
       (yt-dlp, FFmpeg, chunking)
                  |
                  v
           Speech-to-Text
      Faster-Whisper / Sarvam AI
                  |
             Transcript
                  |
        +---------+----------+
        |                    |
        v                    v
   Gemini Summary      Gemini Insight Extraction
        |                    |
        +---------+----------+
                  |
                  v
       Embeddings + ChromaDB
                  |
                  v
          RAG Question Answering
                  |
                  v
           Streamlit UI
```

## Tech Stack

| Area | Technologies |
|---|---|
| Frontend | Streamlit |
| Language | Python 3.11 |
| Speech-to-text | Faster-Whisper, Sarvam AI |
| LLM | Google Gemini via LangChain |
| RAG framework | LangChain |
| Vector database | ChromaDB |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Audio/video processing | FFmpeg, yt-dlp, pydub |
| Containerization | Docker, Docker Compose |

## Project Structure

```text
AI_Video_RAG/
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
├── utils/
│   └── audio_processor.py
├── downloads/       # Runtime audio-processing files
├── input/           # Optional local input videos
├── model_cache/     # Downloaded ML model files
└── vector_db/       # Persistent ChromaDB data
```

Runtime directories are created or populated as the application runs. Generated media, model files, vector database data, virtual environments, and secrets are excluded from Git.

## Requirements

- Python 3.11
- FFmpeg
- A Google Gemini API key
- A Sarvam AI API key for Hinglish transcription
- Internet access for API calls and initial model downloads

For the Docker setup, install Docker Desktop with the WSL 2 backend on Windows, or Docker Engine and the Compose plugin on Linux.

## Setup: Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SUNRAKU-007/AI_Video_RAG.git
cd AI_Video_RAG
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg separately and make sure the `ffmpeg` command is available on your system PATH.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=low

# Required when using Hinglish transcription
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5

WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=8
WHISPER_BEAM_SIZE=3

SUMMARY_MAX_WORKERS=1
EXTRACTION_MAX_WORKERS=1

AUDIO_WORK_DIR=downloads
CHROMA_DIR=vector_db
COLLECTION_PREFIX=meeting
COLLECTION_NAME=meeting_default
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

Replace the API-key placeholders with your own keys. Never commit `.env` or publish API keys.

### 5. Start the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Setup: Run with Docker

1. Create the `.env` file in the project root using the configuration shown above.
2. Build the image and start the application:

```bash
docker compose up -d --build
```

3. Open [http://localhost:8501](http://localhost:8501).
4. View container logs when troubleshooting:

```bash
docker compose logs -f
```

5. Stop the application:

```bash
docker compose down
```

The Compose configuration mounts `downloads/`, `vector_db/`, and `model_cache/` from the host so runtime files, the vector database, and downloaded model caches persist when the container is recreated. The `input/` directory is mounted read-only.

## Notes

- The first run may take longer because speech and embedding models need to be downloaded and initialized.
- Processing time depends on video duration, hardware, and network/API response times. CPU-based transcription can take longer than the video's runtime.
- Keep API keys in `.env`. The repository's `.gitignore` excludes this file and local runtime data.
- For local videos used through Docker, place the file in the project's `input/` directory and select its container path under `/app/input/`.

## License

No license has been added yet. All rights remain with the repository owner unless a license is added.
