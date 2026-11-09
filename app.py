import streamlit as st
import tempfile
import os
import json

from pathlib import Path

from main import run_pipeline
from core.rag_engine import ask_question


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# PIPELINE CONFIGURATION
# ==================================================

PIPELINE_STAGES = [
    "Audio Processing",
    "Audio Transcription",
    "Title Generation",
    "Transcript Summarization",
    "Action Item Extraction",
    "Key Decision Extraction",
    "Open Question Extraction",
    "RAG Pipeline Initialization"
]


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 10% 5%,
                #192442 0%,
                transparent 30%
            ),
            #0b1020;
        color: #f1f5f9;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #10182b;
        border-right: 1px solid #27334b;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(
            120deg,
            #172554,
            #312e81,
            #4c1d95
        );
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        padding: 18px;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #27334b;
    }

    .stTabs [data-baseweb="tab"] {
        background: #151f35;
        border-radius: 10px 10px 0 0;
        padding: 12px 20px;
        color: #cbd5e1;
    }

    .stTabs [aria-selected="true"] {
        background: #312e81;
        color: white;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #6366f1;
        transition: 0.2s;
    }

    .stButton > button:hover {
        border-color: #a5b4fc;
        color: white;
    }

    .stTextInput input,
    .stTextArea textarea {
        background-color: #151f35;
        color: white;
        border: 1px solid #334155;
        border-radius: 10px;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .muted-text {
        color: #94a3b8;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def convert_to_text(value):

    if value is None:
        return "No information available."

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            indent=2,
            ensure_ascii=False
        )

    return str(value)


def render_content(value):

    if value is None:
        st.info("No information available.")
        return

    if isinstance(value, str) and not value.strip():
        st.info("No information available.")
        return

    if isinstance(value, list):

        for item in value:
            st.markdown(f"- {item}")

    elif isinstance(value, dict):

        st.json(value)

    else:

        st.markdown(str(value))


def count_items(value):

    if value is None:
        return 0

    if isinstance(value, (list, dict)):
        return len(value)

    if isinstance(value, str):

        return len([
            line for line in value.splitlines()
            if line.strip()
        ])

    return 0


def format_duration(seconds):

    if not isinstance(seconds, (int, float)):
        return str(seconds)

    if seconds < 60:
        return f"{seconds:.1f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes} min {remaining_seconds:.1f} sec"


# ==================================================
# SESSION STATE
# ==================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🎬 AI Video Assistant")

    st.caption(
        "Your personal video intelligence workspace"
    )

    st.divider()

    st.markdown("### 📁 Input Source")

    source_mode = st.radio(
        "Choose input type",
        [
            "YouTube URL",
            "Upload a file",
            "Local file path"
        ]
    )

    source_url = ""
    uploaded_file = None
    local_path = ""

    if source_mode == "YouTube URL":

        source_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=..."
        )

    elif source_mode == "Upload a file":

        uploaded_file = st.file_uploader(
            "Upload audio/video",
            type=[
                "mp3",
                "mp4",
                "wav",
                "m4a",
                "webm",
                "mkv",
                "mov",
                "avi",
                "flac",
                "ogg",
                "aac"
            ],
            help="Upload a supported audio or video file."
        )

    elif source_mode == "Local file path":

        local_path = st.text_input(
            "File path",
            placeholder=r"C:\Users\YourName\Videos\video.mp4",
            help=(
                "Enter a path visible to the application. "
                "When using Docker, use /app/input/filename.mp4."
            )
        )

        st.caption(
            "Docker users: place local files in the project's input folder."
        )

    st.divider()

    st.markdown("### 🌐 Transcription Language")

    language = st.selectbox(
        "Select language",
        ["english", "hinglish"]
    )

    st.markdown("")

    process_clicked = st.button(
        "🚀 Generate Insights",
        use_container_width=True,
        type="primary"
    )

    clear_clicked = st.button(
        "🗑️ Clear Workspace",
        use_container_width=True
    )

    st.divider()

    st.markdown(
        """
        <div class="muted-text">
            Powered by AI transcription, summarization,
            insight extraction and RAG.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# CLEAR WORKSPACE
# ==================================================

if clear_clicked:

    st.session_state.result = None
    st.session_state.chat_history = []

    st.rerun()


# ==================================================
# PROCESS INPUT
# ==================================================

if process_clicked:

    source = None
    temp_path = None

    if source_mode == "YouTube URL":

        source = source_url.strip()

        if not source:
            st.sidebar.error("Please enter a YouTube URL.")

    elif source_mode == "Upload a file":

        if uploaded_file is None:

            st.sidebar.error("Please upload a file.")

        else:

            try:

                suffix = Path(uploaded_file.name).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name

                source = temp_path

            except Exception as error:

                st.sidebar.error(
                    f"File error: {error}"
                )

    elif source_mode == "Local file path":

        source = local_path.strip()

        if not source:

            source = None

            st.sidebar.error(
                "Please enter a file path."
            )

        elif not os.path.isfile(source):

            source = None

            st.sidebar.error(
                "The specified file does not exist."
            )

    # --------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------

    if source:

        st.markdown("## ⚙️ Processing Your Video")

        progress_bar = st.progress(
            0,
            text="Preparing pipeline..."
        )

        stage_placeholder = st.empty()

        stage_states = {
            stage: "pending"
            for stage in PIPELINE_STAGES
        }

        def update_progress(stage, status, message):

            if stage in stage_states:
                stage_states[stage] = status

            completed = sum(
                value == "completed"
                for value in stage_states.values()
            )

            percentage = completed / len(PIPELINE_STAGES)

            progress_bar.progress(
                percentage,
                text=f"{completed}/{len(PIPELINE_STAGES)} stages completed"
            )

            with stage_placeholder.container():

                for stage_name in PIPELINE_STAGES:

                    current_status = stage_states[stage_name]

                    if current_status == "completed":
                        icon = "✅"

                    elif current_status == "running":
                        icon = "⏳"

                    elif current_status == "error":
                        icon = "❌"

                    else:
                        icon = "○"

                    st.write(
                        f"{icon} **{stage_name}**"
                    )

                    if (
                        stage_name == stage
                        and status in ("running", "error")
                    ):
                        st.caption(message)

        try:

            with st.spinner(
                "Processing your video. This may take a while..."
            ):

                result = run_pipeline(
                    source=source,
                    language=language,
                    progress_callback=update_progress
                )

            st.session_state.result = result

            # New video means a fresh conversation.
            st.session_state.chat_history = []

            progress_bar.progress(
                1.0,
                text="All processing stages completed!"
            )

            st.success(
                "Video processed successfully!"
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )

            with st.expander("Processing details"):
                st.exception(error)

        finally:

            if temp_path and os.path.exists(temp_path):

                try:
                    os.remove(temp_path)

                except OSError:
                    st.warning(
                        "Could not remove the temporary uploaded file."
                    )


# ==================================================
# HERO SECTION
# ==================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🎬 AI Video Assistant
        </div>

        <div class="hero-subtitle">
            Transform long videos into summaries, actionable insights,
            decisions and interactive conversations.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# EMPTY STATE
# ==================================================

result = st.session_state.result

if result is None:

    st.markdown(
        "## Welcome to your Video Workspace"
    )

    st.write(
        "Upload a video, enter a YouTube URL, or provide "
        "a local file path to get started."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📝")

        st.markdown("**Smart Summaries**")

        st.caption(
            "Generate meaningful summaries from lengthy videos."
        )

    with col2:

        st.markdown("### ✅")

        st.markdown("**Action Extraction**")

        st.caption(
            "Automatically identify tasks, decisions and questions."
        )

    with col3:

        st.markdown("### 💬")

        st.markdown("**Chat with Video**")

        st.caption(
            "Ask questions and retrieve information using RAG."
        )

    st.info(
        "Get started by selecting an input source from the sidebar "
        "and clicking Generate Insights."
    )

    st.stop()


# ==================================================
# DISPLAY PROCESSED VIDEO
# ==================================================

title = result.get("title", "Untitled Video")

transcript = result.get("transcript", "")

summary = result.get("summary", "")

action_items = result.get("action_items", [])

decisions = result.get("key_decisions", [])

questions = result.get("open_questions", [])

timings = result.get("timings", {})


st.subheader(f"📌 {title}")

st.caption(
    "Your video has been successfully processed."
)


# ==================================================
# METRICS
# ==================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "Transcript Words",
        f"{len(str(transcript).split()):,}"
    )

with metric2:

    st.metric(
        "Action Items",
        count_items(action_items)
    )

with metric3:

    st.metric(
        "Key Decisions",
        count_items(decisions)
    )


# ==================================================
# PROCESSING TIME
# ==================================================

if timings:

    total_time = timings.get("total")

    if isinstance(total_time, (int, float)):

        st.success(
            f"⏱️ Total processing time: "
            f"{format_duration(total_time)}"
        )

    with st.expander("View processing breakdown"):

        for stage, duration in timings.items():

            st.write(
                f"**{stage.replace('_', ' ').title()}:** "
                f"{format_duration(duration)}"
            )


st.divider()


# ==================================================
# MAIN TABS
# ==================================================

overview_tab, transcript_tab, chat_tab = st.tabs(
    [
        "📊 Overview & Insights",
        "📄 Transcript",
        "💬 Chat with Video"
    ]
)


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview_tab:

    st.markdown(
        '<div class="section-heading">📝 Video Summary</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        render_content(summary)

    st.markdown("")

    col1, col2 = st.columns(2)

    # ----------------------------------------------
    # ACTION ITEMS
    # ----------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-heading">✅ Action Items</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            render_content(action_items)

    # ----------------------------------------------
    # KEY DECISIONS
    # ----------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-heading">🔑 Key Decisions</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            render_content(decisions)

    # ----------------------------------------------
    # OPEN QUESTIONS
    # ----------------------------------------------

    st.markdown(
        '<div class="section-heading">❓ Open Questions</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        render_content(questions)

    # ----------------------------------------------
    # DOWNLOAD INSIGHTS
    # ----------------------------------------------

    insights_data = {
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions
    }

    insights_text = f"""
VIDEO TITLE
{title}

SUMMARY
{convert_to_text(summary)}

ACTION ITEMS
{convert_to_text(action_items)}

KEY DECISIONS
{convert_to_text(decisions)}

OPEN QUESTIONS
{convert_to_text(questions)}
"""

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="⬇️ Download All Insights",
            data=insights_text,
            file_name="video_insights.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="⬇️ Download Insights JSON",
            data=json.dumps(
                insights_data,
                indent=4,
                ensure_ascii=False
            ),
            file_name="video_insights.json",
            mime="application/json",
            use_container_width=True
        )


# ==================================================
# TAB 2: TRANSCRIPT
# ==================================================

with transcript_tab:

    st.markdown(
        '<div class="section-heading">📄 Complete Transcript</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Here is the complete transcription generated from your video."
    )

    st.text_area(
        "Transcript",
        value=str(transcript),
        height=550,
        label_visibility="collapsed"
    )

    st.download_button(
        label="⬇️ Download Transcript",
        data=str(transcript),
        file_name="video_transcript.txt",
        mime="text/plain",
        use_container_width=True
    )


# ==================================================
# TAB 3: CHAT WITH VIDEO
# ==================================================

with chat_tab:

    st.markdown(
        '<div class="section-heading">💬 Chat with your Video</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Ask questions about the video. Your RAG pipeline will retrieve "
        "relevant information from its transcript."
    )

    if result.get("rag_chain") is None:

        st.warning(
            "The RAG chain is unavailable for this video."
        )

    else:

        # ------------------------------------------
        # CLEAR CHAT HISTORY
        # ------------------------------------------

        col1, col2 = st.columns([5, 1])

        with col2:

            if st.button("Clear Chat"):

                st.session_state.chat_history = []

                st.rerun()

        # ------------------------------------------
        # DISPLAY PREVIOUS MESSAGES
        # ------------------------------------------

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        # ------------------------------------------
        # CHAT INPUT
        # ------------------------------------------

        user_question = st.chat_input(
            "Ask something about your video..."
        )

        if user_question:

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:

                        history_for_request = (
                            st.session_state.chat_history.copy()
                        )

                        history_for_request.append(
                            {
                                "role": "user",
                                "content": user_question
                            }
                        )

                        answer = ask_question(
                            result["rag_chain"],
                            user_question,
                            chat_history=history_for_request
                        )

                        answer = str(answer)

                        st.markdown(answer)

                        st.session_state.chat_history.extend(
                            [
                                {
                                    "role": "user",
                                    "content": user_question
                                },
                                {
                                    "role": "assistant",
                                    "content": answer
                                }
                            ]
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to generate an answer: {error}"
                        )