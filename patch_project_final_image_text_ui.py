
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak_final_ui_patch")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"Backup: {bak}")


def patch_config(project: Path) -> None:
    config_path = project / "config.py"
    if not config_path.exists():
        print("config.py not found; skipping config.py mapping patch.")
        return

    backup(config_path)
    text = config_path.read_text(encoding="utf-8")

    mapping_block = 'MAPPING_OPTIONS = [\n    "Simple Mapping",\n    "RINF_B16",\n]\n'

    if "MAPPING_OPTIONS" in text:
        text = re.sub(
            r"MAPPING_OPTIONS\s*=\s*\[[\s\S]*?\]\s*",
            mapping_block + "\n",
            text,
            count=1,
        )
    else:
        text += "\n\n" + mapping_block

    config_path.write_text(text, encoding="utf-8")
    print("config.py: limited MAPPING_OPTIONS to Simple Mapping / RINF_B16.")


def insert_mapping_override(panels_text: str) -> tuple[str, int]:
    marker = "# FINAL_UI_PATCH_LIMIT_IMAGE_MAPPINGS"
    if marker in panels_text:
        return panels_text, 0

    block = '''
# FINAL_UI_PATCH_LIMIT_IMAGE_MAPPINGS
try:
    MAPPING_OPTIONS = [m for m in MAPPING_OPTIONS if m in ("Simple Mapping", "RINF_B16")]
    if not MAPPING_OPTIONS:
        MAPPING_OPTIONS = ["Simple Mapping", "RINF_B16"]
except NameError:
    MAPPING_OPTIONS = ["Simple Mapping", "RINF_B16"]

'''

    idx = panels_text.find("\ndef ")
    if idx == -1:
        panels_text = block + panels_text
    else:
        panels_text = panels_text[:idx+1] + block + panels_text[idx+1:]
    return panels_text, 1


def patch_hardcoded_mapping_lists(panels_text: str) -> tuple[str, int]:
    n_total = 0
    patterns = [
        r'\[\s*"Simple Mapping"\s*,\s*"R0"[\s\S]*?\]',
        r'\[\s*"SM"\s*,\s*"R0"[\s\S]*?\]',
        r'\[\s*"Simple Mapping"\s*,\s*"RINF_B16"\s*,[\s\S]*?\]',
    ]
    for pat in patterns:
        panels_text, n = re.subn(pat, '["Simple Mapping", "RINF_B16"]', panels_text, count=1)
        n_total += n
    return panels_text, n_total


def patch_side_by_side_preview(panels_text: str) -> tuple[str, int]:
    start_marker = '            preview_file(path, FIELDS["input_preview"])\n\n            if Image is not None:\n'
    start = panels_text.find(start_marker)
    if start == -1:
        return panels_text, 0

    end_marker = '                    st.warning(f"Could not show selected pixel preview: {exc}")'
    end = panels_text.find(end_marker, start)
    if end == -1:
        return panels_text, 0

    end = panels_text.find("\n", end)
    if end == -1:
        end = len(panels_text)

    new_block = '''            # Side-by-side image previews.
            img_col1, img_col2 = st.columns(2, gap="medium")

            with img_col1:
                preview_file(path, FIELDS["input_preview"])

            with img_col2:
                if Image is not None:
                    try:
                        from robust_image_pipeline import prepare_pixel_image
                        im, _meta = prepare_pixel_image(
                            path,
                            pixel_representation=pixel_representation,
                            threshold=int(threshold),
                        )
                        out_dir = WORK_ROOT / "pixel_preview"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        selected_path = out_dir / "selected_pixel_image_panel1.png"
                        im.save(selected_path)
                        preview_file(str(selected_path), "Selected pixel image")
                    except Exception as exc:
                        st.warning(f"Could not show selected pixel preview: {exc}")'''

    panels_text = panels_text[:start] + new_block + panels_text[end:]
    return panels_text, 1


def patch_compression_metrics(panels_text: str) -> tuple[str, int]:
    n = 0

    old_calc = '''        raw_size = int(image_meta.get("raw_pixel_bytes", 0))
        payload_size = len(stored)
        ratio = raw_size / max(1, payload_size)
        saving_pct = (1.0 - payload_size / max(1, raw_size)) * 100.0 if raw_size else 0.0'''

    new_calc = '''        raw_size = int(image_meta.get("raw_pixel_bytes", 0))
        payload_size = len(stored)
        input_file_size = len(st.session_state.get("input_bytes", b"") or b"")
        ratio = raw_size / max(1, payload_size)
        saving_pct = (1.0 - payload_size / max(1, raw_size)) * 100.0 if raw_size else 0.0
        saving_vs_file_pct = (1.0 - payload_size / max(1, input_file_size)) * 100.0 if input_file_size else 0.0
        payload_pct_vs_file = (payload_size / max(1, input_file_size)) * 100.0 if input_file_size else 0.0
        payload_pct_vs_raw = (payload_size / max(1, raw_size)) * 100.0 if raw_size else 0.0'''

    if old_calc in panels_text:
        panels_text = panels_text.replace(old_calc, new_calc, 1)
        n += 1

    old_metrics = '''        c1, c2, c3 = st.columns(3)
        c1.metric("Raw pixel data", fmt_bytes(raw_size))
        if storage_meta.get("method") == "No compression":
            c2.metric("Payload data", fmt_bytes(payload_size))
            c3.metric("Payload ratio", f"{ratio:.2f}×")
            st.caption(
                f"No compression · Pixel representation: {image_meta.get('pixel_representation', pixel_representation)} · "
                f"Payload should match selected raw pixel data."
            )
        else:
            c2.metric("Compressed payload", fmt_bytes(payload_size))
            c3.metric("Compression ratio", f"{ratio:.2f}×")
            st.caption(
                f"Compression saving vs selected raw pixels: {saving_pct:.1f}% · "
                f"Method: {storage_meta.get('method', method)} · "
                f"Level: {image_meta.get('compression_level', compression_level)} · "
                f"Pixel representation: {image_meta.get('pixel_representation', pixel_representation)}"
            )'''

    new_metrics = '''        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Raw pixel data", fmt_bytes(raw_size))
        if storage_meta.get("method") == "No compression":
            c2.metric("Payload data", fmt_bytes(payload_size))
            c3.metric("Payload ratio", f"{ratio:.2f}×")
            c4.metric("↓ vs input file", f"{saving_vs_file_pct:.1f}%")
            c5.metric("↓ vs raw pixels", f"{saving_pct:.1f}%")
        else:
            c2.metric("Compressed payload", fmt_bytes(payload_size))
            c3.metric("Compression ratio", f"{ratio:.2f}×")
            c4.metric("↓ vs input file", f"{saving_vs_file_pct:.1f}%")
            c5.metric("↓ vs raw pixels", f"{saving_pct:.1f}%")'''

    if old_metrics in panels_text:
        panels_text = panels_text.replace(old_metrics, new_metrics, 1)
        n += 1

    old_quality = '''            {"Metric": "Payload data" if storage_meta.get("method") == "No compression" else "Compressed payload", "Value": fmt_bytes(payload_size)},
            {"Metric": "Payload ratio" if storage_meta.get("method") == "No compression" else "Compression ratio", "Value": f"{ratio:.2f}×"},'''

    new_quality = '''            {"Metric": "Payload data" if storage_meta.get("method") == "No compression" else "Compressed payload", "Value": fmt_bytes(payload_size)},
            {"Metric": "Payload ratio" if storage_meta.get("method") == "No compression" else "Compression ratio", "Value": f"{ratio:.2f}×"},
            {"Metric": "Payload vs input file", "Value": f"{payload_pct_vs_file:.2f}% of original file"},
            {"Metric": "Reduction vs input file", "Value": f"{saving_vs_file_pct:.2f}%"},
            {"Metric": "Payload vs raw pixels", "Value": f"{payload_pct_vs_raw:.2f}% of raw pixels"},
            {"Metric": "Reduction vs raw pixels", "Value": f"{saving_pct:.2f}%"},'''

    if old_quality in panels_text:
        panels_text = panels_text.replace(old_quality, new_quality, 1)
        n += 1

    return panels_text, n


def patch_panels(project: Path) -> None:
    panels_path = project / "panels.py"
    if not panels_path.exists():
        print("panels.py not found; skipping image UI patch.")
        return

    backup(panels_path)
    text = panels_path.read_text(encoding="utf-8")

    text, n0 = insert_mapping_override(text)
    text, n1 = patch_hardcoded_mapping_lists(text)
    text, n2 = patch_side_by_side_preview(text)
    text, n3 = patch_compression_metrics(text)

    panels_path.write_text(text, encoding="utf-8")

    print("panels.py patched:")
    print(f"  mapping override inserted: {n0}")
    print(f"  hardcoded mapping list replacements: {n1}")
    print(f"  side-by-side preview replacements: {n2}")
    print(f"  compression metric replacements: {n3}")

    if n2 == 0:
        print("WARNING: Side-by-side preview block not found. Send panels.py if previews remain vertical.")
    if n3 == 0:
        print("WARNING: Compression metric block not found. Send panels.py if metrics did not appear.")


def copy_text_module(project: Path, patch_dir: Path) -> None:
    src = patch_dir / "text_sparse_semantic_dna_streamlit.py"
    if not src.exists():
        print("Bundled text module not found; skipping text module copy.")
        return

    dst = project / "text_sparse_semantic_dna_streamlit.py"
    backup(dst)
    shutil.copy2(src, dst)
    print("Copied updated text_sparse_semantic_dna_streamlit.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", help="Project folder path")
    args = parser.parse_args()

    project = Path(args.project_dir).expanduser().resolve()
    if not project.exists():
        raise FileNotFoundError(project)

    patch_dir = Path(__file__).resolve().parent

    copy_text_module(project, patch_dir)
    patch_config(project)
    patch_panels(project)

    print("\nDone. Run:")
    print(f"  cd {project}")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
