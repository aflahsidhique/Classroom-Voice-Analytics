"""Streamlit demo interface for the Classroom Voice Analytics MVP.

Two modes:
  - Browse a precomputed sample session (instant, no heavy compute -
    what the hosted "Live Demo Link" defaults to).
  - Upload your own short audio clip and run the pipeline live. Live mode
    uses a small Whisper model and caps the processed duration so it stays
    within free-tier hosting limits (see README "System design notes").
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.pipeline import list_cached, load_cached, process_audio_streaming

st.set_page_config(page_title="Classroom Voice Analytics", layout="wide")

PALETTE = {"Teacher": "#2563eb", "Student": "#f59e0b", "Silence": "#cbd5e1"}
LIVE_MODE_MAX_SECONDS = 240  # keep hosted live-inference bounded (~4 min)


def render_result(result: dict):
    metrics = result["metrics"]

    st.subheader("Classroom Summary")
    st.write(result["summary"])

    st.subheader("Engagement Metrics")
    cols = st.columns(len(metrics["metrics"]))
    for col, m in zip(cols, metrics["metrics"]):
        col.metric(m["name"], f"{m['value']} {m['unit'].split(' ')[0]}")
    with st.expander("Formulas & interpretation"):
        for m in metrics["metrics"]:
            st.markdown(
                f"**{m['name']}** — `{m['formula']}`  \n"
                f"{m['explanation']}  \n"
                f"*Interpretation:* {m['interpretation']}"
            )

    st.subheader("Talk-Time Breakdown")
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Teacher", "Student", "Silence"],
                    values=[metrics["teacher_talk_sec"], metrics["student_talk_sec"], metrics["silence_sec"]],
                    marker=dict(colors=[PALETTE["Teacher"], PALETTE["Student"], PALETTE["Silence"]]),
                    hole=0.5,
                )
            ]
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(
            f"- **Teacher turns:** {metrics['teacher_turns']}  \n"
            f"- **Student turns:** {metrics['student_turns']}  \n"
            f"- **Teacher questions:** {metrics['teacher_questions']}  \n"
            f"- **Student responses:** {metrics['student_responses']}  \n"
            f"- **Duration:** {metrics['duration_sec']/60:.1f} min"
        )

    if result.get("highlights"):
        st.subheader("Highlights")
        for h in result["highlights"]:
            st.markdown(f"- {h}")

    st.subheader("Transcript")
    df = pd.DataFrame(result["turns"])
    if not df.empty:
        df["time"] = df["start"].apply(lambda s: f"{int(s//60):02d}:{int(s%60):02d}")
        df["tag"] = df.apply(
            lambda r: ("❓" if r["is_question"] else "") + ("✅" if r["is_response"] else ""), axis=1
        )
        display_df = df[["time", "speaker", "tag", "text"]].rename(columns={"tag": "flags"})

        def highlight_speaker(row):
            color = PALETTE.get(row["speaker"], "#ffffff")
            return [f"background-color: {color}22"] * len(row)

        st.dataframe(display_df.style.apply(highlight_speaker, axis=1), use_container_width=True, height=420)
    else:
        st.info("No speech segments detected.")


def main():
    st.title("🎙️ Classroom Voice Analytics MVP")
    st.caption(
        "Converts classroom audio into a speaker-labeled transcript, teacher/student talk-time "
        "metrics, and an auto-generated classroom summary — fully offline-capable (Whisper + "
        "local speaker clustering, no external API calls)."
    )

    tab_sample, tab_upload = st.tabs(["📁 Sample sessions", "⬆️ Upload your own audio"])

    with tab_sample:
        cached = list_cached()
        if not cached:
            st.warning("No precomputed sample sessions found in outputs/. Run scripts/run_pipeline.py first.")
        else:
            choice = st.selectbox("Choose a precomputed session", cached)
            result = load_cached(choice)
            render_result(result)

    with tab_upload:
        st.info(
            f"Live mode uses the '{config.WHISPER_MODEL_SIZE_FAST}' Whisper model and processes at most "
            f"the first {LIVE_MODE_MAX_SECONDS//60} minutes, to keep inference time reasonable on shared "
            "hosting. Results stream in as the audio is transcribed — you'll see transcript and metrics "
            "within seconds, updating live, rather than waiting for the whole clip. For full-length, "
            "higher-accuracy runs, use scripts/run_pipeline.py locally."
        )
        uploaded = st.file_uploader("Upload a classroom audio clip", type=["mp3", "wav", "m4a", "ogg"])
        language = st.text_input("Force language code (optional, e.g. 'hi'). Leave blank to auto-detect.")
        if uploaded is not None and st.button("Run analysis"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            progress_bar = st.progress(0.0, text="Starting transcription...")
            results_area = st.empty()

            for partial in process_audio_streaming(
                tmp_path,
                model_size=config.WHISPER_MODEL_SIZE_FAST,
                language=language or None,
                max_duration_sec=LIVE_MODE_MAX_SECONDS,
            ):
                total = partial.get("total_duration_sec") or partial["duration_sec"] or 1.0
                frac = min(partial["duration_sec"] / total, 1.0)
                status = (
                    "Done."
                    if partial["is_final"]
                    else f"Transcribed {partial['duration_sec']:.0f}s / {total:.0f}s..."
                )
                progress_bar.progress(frac, text=status)
                with results_area.container():
                    render_result(partial)

            progress_bar.empty()


if __name__ == "__main__":
    main()
