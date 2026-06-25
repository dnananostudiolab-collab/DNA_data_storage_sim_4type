from __future__ import annotations

"""
patch_project_two_branch.py

Apply the two-branch Image/Text DNA Storage update to an existing project.

Usage:
    python patch_project_two_branch.py /path/to/your/project
"""

import argparse
import re
import shutil
from pathlib import Path


APP_CODE = 'from __future__ import annotations\n\n"""\nTwo-branch Streamlit app for DNA Data Storage.\n\nBranch 1 — Image DNA Storage\n    Uses your existing image/compression pipeline and exposes only:\n        - Simple Mapping (SM)\n        - RINF_B16 (R∞)\n\nBranch 2 — Text DNA Storage\n    Uses:\n        text_sparse_semantic_dna_streamlit.render_text_dna_storage_panel\n"""\n\nimport streamlit as st\n\nfrom config import APP_TITLE\nfrom ui_design_system.streamlit_style import apply_app_style\nfrom ui_design_system.ui_labels import PANEL_TITLES, STATUS\n\nfrom panels import (\n    render_panel_1_upload,\n    render_panel_2_compression,\n    render_panel_3_encoding,\n    render_panel_4_experiment,\n    render_panel_5_decoding,\n    render_panel_6_analysis,\n)\n\nfrom text_sparse_semantic_dna_streamlit import render_text_dna_storage_panel\n\n\nAPP_STEPS = [\n    (1, PANEL_TITLES["input"]),\n    (2, PANEL_TITLES["data_encoding"]),\n    (3, PANEL_TITLES["dna_encoding"]),\n    (4, PANEL_TITLES["strand_preparation"]),\n    (5, PANEL_TITLES["file_decoding"]),\n    (6, PANEL_TITLES["validation"]),\n]\n\n\ndef _step_state(step_no: int) -> tuple[str, str]:\n    checks = {\n        1: bool(st.session_state.get("input_bytes")),\n        2: bool(st.session_state.get("stored_bytes")),\n        3: bool(st.session_state.get("dna")),\n        4: bool(st.session_state.get("strand_rows")),\n        5: bool(st.session_state.get("decoded_data")),\n        6: bool(st.session_state.get("restored_info")),\n    }\n    if checks.get(step_no):\n        return "done", STATUS["done"]\n    previous_done = all(checks.get(i) for i in range(1, step_no)) if step_no > 1 else True\n    if previous_done:\n        return "current", "Next"\n    return "", STATUS["waiting"]\n\n\ndef _render_hero() -> None:\n    st.markdown(\n        """\n<div class="hero-card">\n  <div class="hero-title">🧬 DNA Storage Pipeline</div>\n  <div class="hero-subtitle">\n    Two-branch system: image compression with SM/R∞ mapping, and semantic token-based text DNA storage.\n  </div>\n</div>\n""",\n        unsafe_allow_html=True,\n    )\n\n\ndef _render_stepper() -> None:\n    parts = [\'<div class="pipeline-steps">\']\n    for n, label in APP_STEPS:\n        css, state = _step_state(n)\n        parts.append(\n            f\'<div class="pipeline-step {css}">\'\n            f\'<div><span class="step-num">{n}</span><span class="step-name">{label}</span></div>\'\n            f\'<div class="step-state">{state}</div>\'\n            f\'</div>\'\n        )\n    parts.append("</div>")\n    st.markdown("".join(parts), unsafe_allow_html=True)\n\n\ndef render_image_branch() -> None:\n    st.markdown("## Image DNA Storage")\n    st.caption("Image compression branch. DNA mapping is limited to SM and R∞ in config.py.")\n\n    _render_stepper()\n    render_panel_1_upload()\n    render_panel_2_compression()\n    render_panel_3_encoding()\n    render_panel_4_experiment()\n    render_panel_5_decoding()\n    render_panel_6_analysis()\n\n\ndef render_text_branch() -> None:\n    st.markdown("## Text DNA Storage")\n    st.caption(\n        "Sparse semantic token coding: compressed + openable + non-exact readable text recovery under substitution errors."\n    )\n    render_text_dna_storage_panel()\n\n\ndef render_app() -> None:\n    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")\n    apply_app_style()\n    _render_hero()\n\n    tab_image, tab_text = st.tabs(["🖼️ Image DNA Storage", "📝 Text DNA Storage"])\n\n    with tab_image:\n        render_image_branch()\n\n    with tab_text:\n        render_text_branch()\n\n\nif __name__ == "__main__":\n    render_app()\n'


def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak_two_branch")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"Backup: {bak}")


def patch_mapping_options(config_path: Path) -> None:
    if not config_path.exists():
        print("config.py not found; skipping MAPPING_OPTIONS patch.")
        return

    backup(config_path)
    text = config_path.read_text(encoding="utf-8")

    replacement = 'MAPPING_OPTIONS = [\n    "Simple Mapping",\n    "RINF_B16",\n]\n'

    if "MAPPING_OPTIONS" not in text:
        text += "\n\n" + replacement
    else:
        text = re.sub(
            r"MAPPING_OPTIONS\s*=\s*\[[\s\S]*?\]\s*",
            replacement,
            text,
            count=1,
        )

    config_path.write_text(text, encoding="utf-8")
    print("Patched config.py: MAPPING_OPTIONS = ['Simple Mapping', 'RINF_B16']")


def patch_ui_labels(ui_labels_path: Path) -> None:
    if not ui_labels_path.exists():
        print("ui_design_system/ui_labels.py not found; skipping label patch.")
        return

    backup(ui_labels_path)
    text = ui_labels_path.read_text(encoding="utf-8")

    mapping_display = """MAPPING_DISPLAY = {
    "Simple Mapping": "SM",
    "RINF_B16": "R∞",
}
"""
    mapping_order = """MAPPING_ORDER = [
    "Simple Mapping",
    "RINF_B16",
]
"""

    if "MAPPING_DISPLAY" in text:
        text = re.sub(r"MAPPING_DISPLAY\s*=\s*\{[\s\S]*?\}\s*", mapping_display, text, count=1)
    else:
        text += "\n\n" + mapping_display

    if "MAPPING_ORDER" in text:
        text = re.sub(r"MAPPING_ORDER\s*=\s*\[[\s\S]*?\]\s*", mapping_order, text, count=1)
    else:
        text += "\n\n" + mapping_order

    ui_labels_path.write_text(text, encoding="utf-8")
    print("Patched ui_labels.py: display only SM and R∞")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", help="Path to existing Streamlit project folder")
    args = parser.parse_args()

    project = Path(args.project_dir).expanduser().resolve()
    if not project.exists():
        raise FileNotFoundError(project)

    here = Path(__file__).resolve().parent
    text_module = here / "text_sparse_semantic_dna_streamlit.py"

    if not text_module.exists():
        raise FileNotFoundError("text_sparse_semantic_dna_streamlit.py must be next to this patch script.")

    shutil.copy2(text_module, project / "text_sparse_semantic_dna_streamlit.py")
    print("Copied text_sparse_semantic_dna_streamlit.py")

    app_path = project / "app.py"
    backup(app_path)
    app_path.write_text(APP_CODE, encoding="utf-8")
    print("Wrote two-branch app.py")

    patch_mapping_options(project / "config.py")
    patch_ui_labels(project / "ui_design_system" / "ui_labels.py")

    print("\nDone.")
    print("Run:")
    print(f"  cd {project}")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
