from __future__ import annotations

import streamlit as st

from config import APP_TITLE
from ui_design_system.streamlit_style import apply_app_style
from ui_design_system.ui_labels import PANEL_TITLES, STATUS

from panels import (
    render_panel_1_upload,
    render_panel_2_compression,
    render_panel_3_encoding,
    render_panel_4_experiment,
    render_panel_5_decoding,
    render_panel_6_analysis,
)
from text_dna_unified_panel import render_text_dna_storage_panel
from audio_dna_tab import render_audio_dna_storage_panel
from video_dna_tab import render_video_dna_storage_panel


APP_STEPS = [
    (1, PANEL_TITLES["input"]),
    (2, PANEL_TITLES["data_encoding"]),
    (3, PANEL_TITLES["dna_encoding"]),
    (4, PANEL_TITLES["strand_preparation"]),
    (5, PANEL_TITLES["file_decoding"]),
    (6, PANEL_TITLES["validation"]),
]


def _step_state(step_no: int) -> tuple[str, str]:
    checks = {
        1: bool(st.session_state.get("input_bytes")),
        2: bool(st.session_state.get("stored_bytes")),
        3: bool(st.session_state.get("dna")),
        4: bool(st.session_state.get("strand_rows")),
        5: bool(st.session_state.get("decoded_data")),
        6: bool(st.session_state.get("restored_info")),
    }
    if checks.get(step_no):
        return "done", STATUS["done"]
    previous_done = all(checks.get(i) for i in range(1, step_no)) if step_no > 1 else True
    if previous_done:
        return "current", "Next"
    return "", STATUS["waiting"]


def _render_compact_overrides() -> None:
    st.markdown(
        """
<style>
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2.0rem;
    max-width: 1380px;
}
.hero-card {
    border: 1px solid rgba(125, 125, 125, 0.18);
    border-radius: 22px;
    padding: 1.05rem 1.25rem;
    margin: 0.2rem 0 1.0rem 0;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.10), rgba(16, 185, 129, 0.08));
}
.hero-title {
    font-size: 1.55rem;
    font-weight: 760;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    margin-top: 0.25rem;
    opacity: 0.78;
    font-size: 0.95rem;
}
.pipeline-steps {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.55rem;
    margin: 0.4rem 0 0.8rem 0;
}
.pipeline-step {
    border: 1px solid rgba(125, 125, 125, 0.18);
    border-radius: 16px;
    padding: 0.55rem 0.65rem;
    background: rgba(125, 125, 125, 0.045);
}
.pipeline-step.done {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.08);
}
.pipeline-step.current {
    border-color: rgba(99, 102, 241, 0.45);
    background: rgba(99, 102, 241, 0.08);
}
.step-num {
    display: inline-flex;
    width: 1.45rem;
    height: 1.45rem;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    margin-right: 0.35rem;
    background: rgba(125, 125, 125, 0.13);
    font-weight: 700;
    font-size: 0.82rem;
}
.step-name {
    font-weight: 650;
    font-size: 0.86rem;
}
.step-state {
    margin-top: 0.25rem;
    font-size: 0.75rem;
    opacity: 0.68;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(125, 125, 125, 0.14);
    border-radius: 16px;
    padding: 0.65rem 0.75rem;
    background: rgba(125, 125, 125, 0.035);
}
div[data-testid="stVerticalBlock"] { gap: 0.65rem; }
@media (max-width: 1100px) {
    .pipeline-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

/* Compact, consistent controls */
.stButton > button, .stDownloadButton > button {
    width: auto !important;
    min-width: 148px;
    max-width: 260px;
    border-radius: 12px !important;
    padding: 0.42rem 0.85rem !important;
    font-size: 0.88rem !important;
    font-weight: 650 !important;
    border: 1px solid rgba(79, 70, 229, 0.35) !important;
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.92), rgba(14, 165, 233, 0.90)) !important;
    color: white !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.92), rgba(5, 150, 105, 0.90)) !important;
    border-color: rgba(16, 185, 129, 0.38) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.03);
    transform: translateY(-1px);
}
label, .stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label, .stRadio label, .stCheckbox label, .stSlider label {
    font-size: 0.92rem !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
<div class="hero-card">
  <div class="hero-title">DNA Data Storage System</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_stepper() -> None:
    parts = ['<div class="pipeline-steps">']
    for n, label in APP_STEPS:
        css, state = _step_state(n)
        parts.append(
            f'<div class="pipeline-step {css}">'
            f'<div><span class="step-num">{n}</span><span class="step-name">{label}</span></div>'
            f'<div class="step-state">{state}</div>'
            f'</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_image_branch() -> None:
    st.markdown("## Image DNA Storage")
    _render_stepper()
    render_panel_1_upload()
    render_panel_2_compression()
    render_panel_3_encoding()
    render_panel_4_experiment()
    render_panel_5_decoding()
    render_panel_6_analysis()


def render_text_branch() -> None:
    st.markdown("## Text DNA Storage")
    render_text_dna_storage_panel()


def render_audio_branch() -> None:
    render_audio_dna_storage_panel()


def render_video_branch() -> None:
    render_video_dna_storage_panel()


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    apply_app_style()
    _render_compact_overrides()
    _render_hero()

    tab_image, tab_text, tab_audio, tab_video = st.tabs(["🖼️ Image", "📝 Text", "🎧 Audio", "🎬 Video"])

    with tab_image:
        render_image_branch()

    with tab_text:
        render_text_branch()

    with tab_audio:
        render_audio_branch()

    with tab_video:
        render_video_branch()


if __name__ == "__main__":
    render_app()
