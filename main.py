import time
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input, cleanup_processed_audio
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all_insights
from core.rag_engine import build_rag_chain, ask_question


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def notify_progress(
    stage_name,
    status,
    message,
    progress_callback=None
):
    logger.info("[%s] %s: %s", status.upper(), stage_name, message)

    if progress_callback:
        progress_callback(stage_name, status, message)


def run_stage(
    stage_name,
    function,
    *args,
    progress_callback=None
):
    notify_progress(
        stage_name,
        "running",
        f"{stage_name} started.",
        progress_callback
    )

    start = time.perf_counter()

    try:
        result = function(*args)
        elapsed = time.perf_counter() - start

        notify_progress(
            stage_name,
            "completed",
            f"Completed in {elapsed:.2f} seconds.",
            progress_callback
        )

        return result, elapsed

    except Exception as error:
        notify_progress(
            stage_name,
            "error",
            str(error),
            progress_callback
        )

        logger.exception("Stage failed: %s", stage_name)
        raise


def cleanup_audio(chunks):
    if not chunks:
        return

    try:
        cleanup_processed_audio(chunks)
        logger.info("Temporary audio cleaned up.")

    except Exception:
        logger.warning(
            "Audio cleanup failed.",
            exc_info=True
        )


def run_parallel_analysis(
    transcript,
    progress_callback=None
):
    """
    Run summarization and insight extraction concurrently.

    Progress callbacks execute on the main thread.
    """

    stages = {
        "summary": "Transcript Summarization",
        "insights": [
            "Action Item Extraction",
            "Key Decision Extraction",
            "Open Question Extraction"
        ]
    }

    for stage in [stages["summary"]] + stages["insights"]:
        notify_progress(
            stage,
            "running",
            "Analysis started.",
            progress_callback
        )

    results = {}
    timings = {}
    errors = []

    analysis_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:

        futures = {
            executor.submit(summarize, transcript): "summary",
            executor.submit(extract_all_insights, transcript): "insights"
        }

        task_start = {
            "summary": time.perf_counter(),
            "insights": time.perf_counter()
        }

        for future in as_completed(futures):

            task = futures[future]

            elapsed = time.perf_counter() - task_start[task]
            timings[task] = round(elapsed, 2)

            try:
                result = future.result()

                if task == "summary":
                    results["summary"] = result

                    notify_progress(
                        stages["summary"],
                        "completed",
                        f"Completed in {elapsed:.2f} seconds.",
                        progress_callback
                    )

                else:
                    if not isinstance(result, dict):
                        raise TypeError(
                            "Insight extraction must return a dictionary."
                        )

                    results["insights"] = result

                    for stage in stages["insights"]:
                        notify_progress(
                            stage,
                            "completed",
                            f"Completed in {elapsed:.2f} seconds.",
                            progress_callback
                        )

            except Exception as error:
                errors.append((task, error))

                failed_stages = (
                    [stages["summary"]]
                    if task == "summary"
                    else stages["insights"]
                )

                for stage in failed_stages:
                    notify_progress(
                        stage,
                        "error",
                        str(error),
                        progress_callback
                    )

                logger.exception(
                    "Parallel analysis failed: %s",
                    task
                )

    if errors:
        task, error = errors[0]

        raise RuntimeError(
            f"{task} failed: {error}"
        ) from error

    timings["parallel_wall_time"] = round(
        time.perf_counter() - analysis_start,
        2
    )

    return (
        results["summary"],
        results["insights"],
        timings
    )


def run_pipeline(
    source: str,
    language: str = "english",
    progress_callback=None
) -> dict:

    if not source or not str(source).strip():
        raise ValueError("Please provide a valid URL or file path.")

    language = language.lower().strip()

    if language not in ("english", "hinglish"):
        raise ValueError(
            "Language must be either 'english' or 'hinglish'."
        )

    pipeline_start = time.perf_counter()

    timings = {}
    chunks = None

    logger.info("=" * 60)
    logger.info("Starting AI Video Assistant")
    logger.info("=" * 60)

    # ------------------------------------------
    # 1. AUDIO PROCESSING
    # ------------------------------------------

    chunks, elapsed = run_stage(
        "Audio Processing",
        process_input,
        source,
        progress_callback=progress_callback
    )

    timings["audio_processing"] = round(elapsed, 2)

    if not chunks:
        raise RuntimeError("No audio chunks were generated.")

    logger.info("Generated %d audio chunks.", len(chunks))

    # ------------------------------------------
    # 2. TRANSCRIPTION
    # ------------------------------------------

    try:
        transcript, elapsed = run_stage(
            "Audio Transcription",
            transcribe_all,
            chunks,
            language,
            progress_callback=progress_callback
        )

        timings["transcription"] = round(elapsed, 2)

    finally:
        cleanup_audio(chunks)

    if not transcript or not transcript.strip():
        raise RuntimeError("Transcription returned empty text.")

    logger.info(
        "Transcription completed. Characters: %d",
        len(transcript)
    )

    # ------------------------------------------
    # 3. TITLE GENERATION
    # ------------------------------------------

    title, elapsed = run_stage(
        "Title Generation",
        generate_title,
        transcript,
        progress_callback=progress_callback
    )

    timings["title_generation"] = round(elapsed, 2)

    # ------------------------------------------
    # 4. SUMMARY + INSIGHT EXTRACTION
    # ------------------------------------------

    summary, insights, analysis_timings = run_parallel_analysis(
        transcript,
        progress_callback=progress_callback
    )

    timings.update({
        "summarization": analysis_timings["summary"],
        "insight_extraction": analysis_timings["insights"],
        "parallel_analysis_wall_time": analysis_timings["parallel_wall_time"]
    })

    action_items = insights.get("action_items", "")
    key_decisions = insights.get("key_decisions", "")
    open_questions = insights.get("open_questions", "")

    # ------------------------------------------
    # 5. RAG INITIALIZATION
    # ------------------------------------------

    rag_chain, elapsed = run_stage(
        "RAG Pipeline Initialization",
        build_rag_chain,
        transcript,
        progress_callback=progress_callback
    )

    timings["rag_initialization"] = round(elapsed, 2)

    # ------------------------------------------
    # TOTAL TIME
    # ------------------------------------------

    timings["total"] = round(
        time.perf_counter() - pipeline_start,
        2
    )

    logger.info("=" * 60)
    logger.info(
        "Pipeline completed in %.2f seconds.",
        timings["total"]
    )
    logger.info("=" * 60)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
        "timings": timings
    }


# ==================================================
# CLI
# ==================================================

if __name__ == "__main__":

    print("\n🎬 AI Video Assistant")
    print("=" * 60)

    source = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    language = input(
        "Language (english/hinglish): "
    ).strip().lower() or "english"

    try:

        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print("📌 TITLE")
        print(result["title"])

        print("\n📋 SUMMARY")
        print(result["summary"])

        print("\n✅ ACTION ITEMS")
        print(result["action_items"])

        print("\n🔑 KEY DECISIONS")
        print(result["key_decisions"])

        print("\n❓ OPEN QUESTIONS")
        print(result["open_questions"])

        print("\n⏱️ PROCESSING TIMES")

        for stage, duration in result["timings"].items():
            print(f"{stage}: {duration} seconds")

        print("\n" + "=" * 60)
        print("💬 Chat with your video. Type 'exit' to quit.")

        while True:

            question = input("\nYou: ").strip()

            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                answer = ask_question(
                    result["rag_chain"],
                    question
                )

                print(f"\nAssistant: {answer}")

            except Exception as error:
                print(f"Unable to answer: {error}")

    except Exception as error:
        logger.exception("Pipeline failed.")
        print(f"\nPipeline failed: {error}")