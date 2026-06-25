from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from compression_pipeline import CompressionCandidate, run_compression_benchmark
from robust_image_pipeline import (
    list_image_compression_methods,
    list_compression_levels,
    list_pixel_representations,
    raw_pixel_info,
    encode_image_to_payload,
    decode_payload_to_image,
    method_summary,
    method_caption,
)
from config import WORK_ROOT, MAPPING_OPTIONS, DNA_PREVIEW_HEIGHT
from dna_codec import gc_content, homopolymer_stats
from dna_mapping import decode_dna_with_mapping, encode_bytes_to_dna, validate_container
from fragments import clean_dna, choose_auto_strand_design, prepare_dna_strands, strand_rows_to_csv
from restore_analysis import image_metrics, text_similarity, write_restored_file
from toolkit_adapter import reconstruct_consensus_from_reads, simulate_sequencing_reads
from ui_helpers import download_bytes_button, fmt_bytes, get_domain, magic_dict, preview_file, save_upload, step_header
from utils_bits_v2 import detect_magic, bytes_to_bitstring
from ui_design_system.ui_labels import (
    PANEL_TITLES, BUTTONS, METRICS, TABS, DATA_SOURCES, FIELDS, MESSAGES, DOWNLOAD_FILES, display_mapping
)
from ui_design_system.design_tokens import REGION_COLORS


# -----------------------------------------------------------------------------
# Small shared helpers.  Keep this file intentionally simple: each helper below
# is used by exactly one or more visible pipeline panels.
# -----------------------------------------------------------------------------


def _preview_seq(seq: str, n: int = 80) -> str:
    seq = clean_dna(seq)
    return seq[:n] + ("..." if len(seq) > n else "")


def _dna_distance(a: str, b: str) -> int:
    """Hamming-style distance including length difference for DNA strings."""
    a = clean_dna(a)
    b = clean_dna(b)
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def _dna_accuracy(a: str, b: str) -> float:
    """Accuracy between two DNA strings; 1.0 means exact recovery."""
    a = clean_dna(a)
    b = clean_dna(b)
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return 1.0 - (_dna_distance(a, b) / denom)


def _candidate_file(cand: CompressionCandidate) -> str:
    out_dir = WORK_ROOT / "selected_compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_method = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in cand.method)
    path = out_dir / f"selected_{safe_method}{cand.ext or '.bin'}"
    path.write_bytes(cand.data)
    return str(path)



def _strand_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    keep = [
        "No.", "Type", "Group", "Shard", "Toolkit column", "Toolkit column index",
        "Strand index", "Index length", "Payload length", "RS parity length",
        "Filler length", "Total length", "GC content", "Longest homopolymer",
    ]
    out = []
    for row in rows:
        out.append({k: row.get(k, "—") for k in keep if k in row})
    return pd.DataFrame(out)


def _reads_table(reads: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["Read ID", "Source No.", "Copy No.", "Read length", "Substitution count", "Insertion count", "Deletion count", "Error count", "Event preview"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in reads])


def _error_events_table(reads: List[Dict[str, Any]]) -> pd.DataFrame:
    events: List[Dict[str, Any]] = []
    for read in reads:
        try:
            for ev in json.loads(read.get("Error events", "[]") or "[]"):
                events.append({
                    "Read ID": ev.get("read_id", read.get("Read ID", "")),
                    "Source strand": ev.get("source_no", read.get("Source No.", "")),
                    "Copy": ev.get("copy_no", read.get("Copy No.", "")),
                    "Original position": ev.get("position_original", ""),
                    "Read position": ev.get("position_read", ""),
                    "Error type": ev.get("operation", ""),
                    "Original base": ev.get("from_base", ""),
                    "Error base": ev.get("to_base", ev.get("inserted_base", "")),
                })
        except Exception:
            continue
    return pd.DataFrame(events)


def _reconstruction_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["No.", "Type", "Toolkit column", "Reads used", "Total length", "Consensus full mismatches", "Consensus payload mismatches"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols if c in r} for r in rows])


def _display_mapping(mapping: str) -> str:
    return display_mapping(mapping)



def _dna_from_strand_rows_for_decode(rows: List[Dict[str, Any]], mapping: str, original_dna_len: int = 0) -> str:
    """Build a decode-ready DNA string from designed or error strand rows."""
    if not rows:
        return ""

    is_rs = mapping == "Reed-Solomon"
    parts: List[str] = []

    if is_rs:
        for row in rows:
            if row.get("Advanced error source"):
                core = clean_dna(row.get("Error payload", ""))
                if core:
                    parts.append(core)
                    continue
            core = clean_dna(row.get("Core strand", ""))
            if not core:
                core = clean_dna(
                    str(row.get("Index", row.get("Strand index", ""))) +
                    str(row.get("Payload", "")) +
                    str(row.get("RS parity", ""))
                )
            if not core:
                full = clean_dna(row.get("Full strand", row.get("Error full strand", "")))
                fbr_len = len(clean_dna(row.get("FBR", "")))
                rbr_len = len(clean_dna(row.get("RBR", "")))
                core = full[fbr_len:len(full) - rbr_len] if rbr_len else full[fbr_len:]
            parts.append(core)
        return clean_dna("".join(parts))

    for row in rows:
        if row.get("Advanced error source"):
            payload = clean_dna(row.get("Error payload", "")) or clean_dna(row.get("Payload", ""))
        else:
            payload = clean_dna(row.get("Payload", ""))
        parts.append(payload)

    dna = clean_dna("".join(parts))
    if original_dna_len:
        dna = dna[:int(original_dna_len)]
    return dna


def _decode_source() -> Tuple[str, str]:
    """Return (label, dna_text) for the selected image reconstruction source."""
    choices: List[str] = ["Original encoded data", "Noisy encoded data", "Uploaded strands CSV"]

    default_source = "Noisy encoded data" if st.session_state.get("noisy_dna") else "Original encoded data"
    previous = st.session_state.get("reconstruction_source", default_source)
    if previous not in choices:
        previous = choices[0]

    label = st.radio(
        "Reconstruction source",
        choices,
        horizontal=True,
        index=choices.index(previous),
        key="decode_source_standard",
    )
    st.session_state["reconstruction_source"] = label

    if label == "Noisy encoded data":
        dna = st.session_state.get("noisy_dna", "")
    elif label == "Uploaded strands CSV":
        dna = st.session_state.get("uploaded_decode_dna", "")
    else:
        dna = st.session_state.get("dna", "")
    return label, dna


def _bytes_to_bit_text(data: bytes) -> str:
    return bytes_to_bitstring(data or b"")


def _download_text_button(label: str, text: str, file_name: str) -> None:
    st.download_button(label, data=text.encode("utf-8"), file_name=file_name, mime="text/plain", use_container_width=True)


def _df_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _strand_rows_from_uploaded_csv(uploaded_file) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), dtype=str).fillna("")
    return [{str(k): str(v) for k, v in row.items()} for row in df.to_dict("records")]


def _clear_downstream_from_storage() -> None:
    for key in [
        "compression_candidates", "selected_candidate", "stored_bytes", "stored_file_path",
        "storage_method", "storage_kind", "storage_meta",
        "dna", "bits", "codec_meta", "strand_rows", "advanced_error_rows", "noisy_dna",
        "dna_error_stats", "reconstruction_source", "reads",
        "wetlab_metrics", "reconstructed_rows", "reconstructed_dna", "reconstruction_metrics",
        "decoded_payload_bytes", "decoded_data", "decoded_raw_pixels", "decoded_bits", "decoded_meta", "decoded_magic", "decoded_valid",
        "decoded_note", "raw_restore_info", "restored_info", "decode_error",
    ]:
        st.session_state.pop(key, None)


def _validate_and_write(data: bytes, preferred: str = "restored") -> Dict[str, Any]:
    out_dir = WORK_ROOT / "decode_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_restored_file(data, str(out_dir), preferred_name=preferred)


def _is_uploaded_image(path: str, data: bytes) -> bool:
    """Return True when the uploaded file can be handled as an image."""
    if Image is None or not data:
        return False
    try:
        domain = get_domain(path, data)
        if domain == "image":
            return True
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def _image_pixels_to_bytes(data: bytes, representation: str, threshold: int = 128) -> Tuple[bytes, Dict[str, Any], bytes]:
    """
    Convert an uploaded image to raw pixel bytes for no-compression storage.

    Returns: (pixel_bytes, metadata, preview_png_bytes).
    The bytes are not an image container; width/height/mode metadata are required
    to reconstruct them later.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image pixel conversion.")
    img = Image.open(io.BytesIO(data))
    if representation == "RGB pixels":
        out_img = img.convert("RGB")
        channels = 3
        raw_mode = "RGB"
        rep_label = "RGB pixels"
    elif representation == "Grayscale pixels":
        out_img = img.convert("L")
        channels = 1
        raw_mode = "L"
        rep_label = "Grayscale pixels"
    elif representation == "Binary image pixels":
        gray = img.convert("L")
        out_img = gray.point(lambda p: 255 if p >= int(threshold) else 0).convert("L")
        channels = 1
        raw_mode = "L"
        rep_label = "Binary image pixels"
    else:
        raise ValueError(f"Unknown image representation: {representation}")

    raw = out_img.tobytes()
    png = io.BytesIO()
    out_img.save(png, format="PNG")
    meta = {
        "kind": "raw_image_pixels",
        "representation": rep_label,
        "raw_mode": raw_mode,
        "width": int(out_img.width),
        "height": int(out_img.height),
        "channels": int(channels),
        "expected_bytes": int(len(raw)),
        "threshold": int(threshold),
        "output_ext": ".png",
    }
    return raw, meta, png.getvalue()


def _raw_image_bytes_to_png(data: bytes, meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Build a PNG preview/output from decoded raw image pixel bytes."""
    if Image is None:
        raise RuntimeError("Pillow is required to restore raw image pixels.")
    width = int(meta.get("width", 0))
    height = int(meta.get("height", 0))
    mode = str(meta.get("raw_mode", "L"))
    expected = int(meta.get("expected_bytes", width * height * (3 if mode == "RGB" else 1)))
    raw = bytes(data or b"")
    note = "Exact raw-pixel length."
    if len(raw) < expected:
        raw = raw + bytes(expected - len(raw))
        note = f"Decoded bytes were shorter than expected; padded {expected - len(data or b'')} bytes."
    elif len(raw) > expected:
        raw = raw[:expected]
        note = f"Decoded bytes were longer than expected; truncated {len(data or b'') - expected} bytes."
    img = Image.frombytes(mode, (width, height), raw)
    png = io.BytesIO()
    img.save(png, format="PNG")
    return png.getvalue(), {"note": note, "width": width, "height": height, "mode": mode, "expected_bytes": expected}


# -----------------------------------------------------------------------------
# DNA Strand Prep visualization + advanced error helper
# -----------------------------------------------------------------------------

_REGION_COLORS = REGION_COLORS


def _row_regions(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return ordered strand regions for normal strands or RS direct strands."""
    is_rs = str(row.get("RS direct", row.get("Toolkit RS direct", ""))).strip().lower() in {"1", "true", "yes", "y"}
    if is_rs or row.get("RS parity"):
        return [
            ("FBR", clean_dna(row.get("FBR", ""))),
            ("Index", clean_dna(row.get("Index", row.get("Strand index", "")))),
            ("Payload", clean_dna(row.get("Payload", ""))),
            ("RS parity", clean_dna(row.get("RS parity", ""))),
            ("RBR", clean_dna(row.get("RBR", ""))),
        ]
    return [
        ("FBR", clean_dna(row.get("FBR", ""))),
        ("SI", clean_dna(row.get("Strand index", ""))),
        ("Payload", clean_dna(row.get("Payload", ""))),
        ("Filler", clean_dna(row.get("Filler", ""))),
        ("RBR", clean_dna(row.get("RBR", ""))),
    ]


def _region_for_position(row: Dict[str, Any], pos1: int) -> str:
    """Return region name for a 1-indexed position in the full strand."""
    cursor = 1
    for name, seq in _row_regions(row):
        end = cursor + len(seq) - 1
        if cursor <= int(pos1) <= end:
            return name
        cursor = end + 1
    return "Outside"


def _region_html(name: str, seq: str, error_positions: set[int] | None = None, *, start_pos: int = 1) -> str:
    """Render one region with optional red marking at 1-indexed full-strand positions."""
    bg, fg = _REGION_COLORS.get(name, ("#f8fafc", "#0f172a"))
    error_positions = error_positions or set()
    chars = []
    for i, ch in enumerate(clean_dna(seq), start=start_pos):
        if i in error_positions:
            ebg, efg = _REGION_COLORS["Error"]
            chars.append(f'<span class="error-base">{ch}</span>')
        else:
            chars.append(ch)
    body = "".join(chars) if chars else "—"
    return (
        f'<span class="region-tag" style="background:{bg};color:{fg};">'
        f'<b>{name}</b>: {body}</span>'
    )


def _render_segmented_strand(row: Dict[str, Any], title: str, *, error_positions: set[int] | None = None) -> None:
    """Show FBR/SI/Payload/Filler/RBR as colored chunks."""
    parts = []
    cursor = 1
    for name, seq in _row_regions(row):
        parts.append(_region_html(name, seq, error_positions, start_pos=cursor))
        cursor += len(seq)
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def _mutate_prepared_strand(
    row: Dict[str, Any],
    *,
    scope: str = "Payload only",
    substitution_rate: float = 0.0,
    insertion_rate: float = 0.0,
    deletion_rate: float = 0.0,
    seed: int = 1,
    allow_indels: bool = False,
) -> Dict[str, Any]:
    """Create one advanced-error strand row and keep exact event positions for UI marking."""
    rng = random.Random(str(seed))
    full = clean_dna(row.get("Full strand", ""))
    if not full:
        full = "".join(seq for _, seq in _row_regions(row))

    is_rs_direct = str(row.get("RS direct", row.get("Toolkit RS direct", ""))).strip().lower() in {"1", "true", "yes", "y"}
    if is_rs_direct:
        mutable_regions = {
            "Payload only": {"Payload"},
            "Index + Payload": {"Index", "Payload"},
            "Full strand": {"FBR", "Index", "Payload", "RS parity", "RBR"},
        }.get(scope, {"Payload"})
    else:
        mutable_regions = {
            "Payload only": {"Payload"},
            "Index + Payload": {"SI", "Payload"},
            "Full strand": {"FBR", "SI", "Payload", "Filler", "RBR"},
        }.get(scope, {"Payload"})

    out: List[str] = []
    events: List[Dict[str, Any]] = []
    sub_count = ins_count = del_count = 0
    read_pos = 0

    for pos, base in enumerate(full, start=1):
        region = _region_for_position(row, pos)
        mutable = region in mutable_regions

        if mutable and allow_indels and rng.random() < float(deletion_rate):
            del_count += 1
            events.append({
                "source_no": row.get("No.", ""),
                "position_original": pos,
                "position_read": read_pos + 1,
                "region": region,
                "operation": "deletion",
                "from_base": base,
                "to_base": "",
            })
            if rng.random() < float(insertion_rate):
                nb = rng.choice([b for b in "ACGT"])
                out.append(nb)
                read_pos += 1
                ins_count += 1
                events.append({
                    "source_no": row.get("No.", ""),
                    "position_original": pos,
                    "position_read": read_pos,
                    "region": region,
                    "operation": "insertion",
                    "from_base": "",
                    "to_base": nb,
                })
            continue

        new_base = base
        if mutable and rng.random() < float(substitution_rate):
            choices = [b for b in "ACGT" if b != base]
            new_base = rng.choice(choices)
            sub_count += 1
            events.append({
                "source_no": row.get("No.", ""),
                "position_original": pos,
                "position_read": read_pos + 1,
                "region": region,
                "operation": "substitution",
                "from_base": base,
                "to_base": new_base,
            })
        out.append(new_base)
        read_pos += 1

        if mutable and allow_indels and rng.random() < float(insertion_rate):
            nb = rng.choice([b for b in "ACGT"])
            out.append(nb)
            read_pos += 1
            ins_count += 1
            events.append({
                "source_no": row.get("No.", ""),
                "position_original": pos,
                "position_read": read_pos,
                "region": region,
                "operation": "insertion",
                "from_base": "",
                "to_base": nb,
            })

    err_full = "".join(out)

    # Reference payload for read reconstruction.  If Advanced Add Errors is used
    # as the wet-lab input, the reconstruction step should measure whether it
    # can recover this error strand from noisy reads.  It should not compare
    # against the clean payload from before Advanced Add Errors, otherwise the
    # reported payload accuracy becomes a fixed "advanced-error accuracy" and
    # appears stuck around e.g. 0.95 for every read-noise setting.
    fbr_len = len(clean_dna(row.get("FBR", "")))
    idx_len = len(clean_dna(row.get("Strand index", "")))
    payload_len = len(clean_dna(row.get("Payload", "")))
    start = fbr_len + idx_len
    err_payload_ref = clean_dna(err_full[start:start + payload_len])
    if is_rs_direct:
        rbr_len = len(clean_dna(row.get("RBR", "")))
        err_payload_ref = clean_dna(err_full[fbr_len:len(err_full) - rbr_len]) if rbr_len else clean_dna(err_full[fbr_len:])

    new = dict(row)
    new.update({
        "Clean full strand": full,
        "Clean payload": clean_dna(row.get("Payload", "")),
        "Error full strand": err_full,
        "Error payload": err_payload_ref,
        "Wet-lab reference payload": err_payload_ref,
        "Full strand": err_full,
        "Advanced error source": "true",
        "Advanced error scope": scope,
        "Advanced error events": json.dumps(events, ensure_ascii=False),
        "Substitution count": str(sub_count),
        "Insertion count": str(ins_count),
        "Deletion count": str(del_count),
        "Error count": str(sub_count + ins_count + del_count),
        "Error full length": str(len(err_full)),
    })

    # If there are no indels, region boundaries are unchanged.  Store mutated
    # region fields so tables and reconstruction references match the actual
    # wet-lab input strand.  With indels, keep the original region fields and use
    # Error full strand + Wet-lab reference payload for recovery/debugging.
    if ins_count == 0 and del_count == 0:
        cursor = 0
        mutated_regions: Dict[str, str] = {}
        for name, seq in _row_regions(row):
            n = len(clean_dna(seq))
            mutated_regions[name] = err_full[cursor:cursor + n]
            cursor += n
        new["FBR"] = mutated_regions.get("FBR", new.get("FBR", ""))
        new["Strand index"] = mutated_regions.get("SI", mutated_regions.get("Index", new.get("Strand index", "")))
        new["Index"] = mutated_regions.get("Index", new.get("Index", new.get("Strand index", "")))
        new["Payload"] = mutated_regions.get("Payload", new.get("Payload", ""))
        new["Filler"] = mutated_regions.get("Filler", new.get("Filler", ""))
        new["RBR"] = mutated_regions.get("RBR", new.get("RBR", ""))
        new["RS parity"] = mutated_regions.get("RS parity", new.get("RS parity", ""))

    return new


def _advanced_error_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["No.", "Advanced error scope", "Total length", "Error full length", "Substitution count", "Insertion count", "Deletion count", "Error count"]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


def _advanced_error_events_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            events = json.loads(row.get("Advanced error events", "[]") or "[]")
        except Exception:
            events = []
        for ev in events:
            out.append({
                "Strand": ev.get("source_no", row.get("No.", "")),
                "Region": ev.get("region", ""),
                "Original position": ev.get("position_original", ""),
                "Error position": ev.get("position_read", ""),
                "Operation": ev.get("operation", ""),
                "Original base": ev.get("from_base", ""),
                "New/inserted base": ev.get("to_base", ""),
            })
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Panel 1 — Upload
# -----------------------------------------------------------------------------




def render_panel_1_upload() -> None:
    with st.container(border=True):
        step_header(1, PANEL_TITLES["input"])
        left, right = st.columns(2, gap="large")

        with left:
            uploaded = st.file_uploader("", type=None, key="upload_input_file")
            if uploaded is not None:
                data_now = uploaded.getvalue()
                upload_sig = f"{uploaded.name}|{len(data_now)}|{hashlib.sha256(data_now).hexdigest()}"
                if st.session_state.get("upload_signature") != upload_sig:
                    path, data = save_upload(uploaded)
                    st.session_state.update({
                        "upload_signature": upload_sig,
                        "input_path": path,
                        "input_bytes": data,
                        "input_name": os.path.basename(path),
                    })
                    _clear_downstream_from_storage()
                elif not st.session_state.get("input_bytes"):
                    path, data = save_upload(uploaded)
                    st.session_state.update({
                        "input_path": path,
                        "input_bytes": data,
                        "input_name": os.path.basename(path),
                    })

        with right:
            data = st.session_state.get("input_bytes")
            path = st.session_state.get("input_path")
            if not data or not path:
                st.info(MESSAGES["upload_to_start"])
                return

            if not _is_uploaded_image(path, data):
                st.error("This pipeline supports image files only.")
                return

            st.markdown("#### Uploaded file")
            m = magic_dict(data)
            f1, f2, f3 = st.columns(3)
            f1.metric("File type", m.get("kind", "unknown"))
            f2.metric("File extension", os.path.splitext(path)[1].lower() or "—")
            f3.metric("Uploaded size", fmt_bytes(len(data)))

            st.markdown("#### Pixel representation")
            prev_pixel = st.session_state.get("pixel_representation", "Grayscale (8 bits/pixel)")
            options = list_pixel_representations()
            if prev_pixel not in options:
                prev_pixel = "Grayscale (8 bits/pixel)"
            pixel_representation = st.selectbox(
                "Pixel representation",
                options,
                index=options.index(prev_pixel),
                key="pixel_representation_select",
            )
            threshold = 128
            if pixel_representation == "Black-white (1 bit/pixel)":
                threshold = st.slider("Black-white threshold", 0, 255, int(st.session_state.get("bw_threshold", 128)), 1, key="bw_threshold_slider")
            st.session_state["pixel_representation"] = pixel_representation
            st.session_state["bw_threshold"] = int(threshold)

            try:
                pix = raw_pixel_info(path, pixel_representation=pixel_representation, threshold=int(threshold))
            except Exception:
                pix = {
                    "width": 0,
                    "height": 0,
                    "channels": 0,
                    "raw_pixel_bytes": 0,
                    "raw_pixel_bits": 0,
                    "bits_per_pixel": 0,
                    "pixel_representation": pixel_representation,
                }

            st.markdown("#### Raw pixel data")
            r1, r2, r3 = st.columns(3)
            r1.metric("Image size", f"{pix.get('width', 0)} × {pix.get('height', 0)}")
            r2.metric("Pixel depth", f"{pix.get('bits_per_pixel', 0)} bit/pixel")
            r3.metric("Raw pixel data", fmt_bytes(pix.get("raw_pixel_bytes", 0)))

            selected_preview_path = ""
            bit_text = ""
            preview_error = ""

            if Image is not None:
                try:
                    from robust_image_pipeline import prepare_pixel_image
                    im, _meta = prepare_pixel_image(path, pixel_representation=pixel_representation, threshold=int(threshold))
                    out_dir = WORK_ROOT / "pixel_preview"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    selected_path = out_dir / "selected_pixel_image_panel1.png"
                    im.save(selected_path)
                    selected_preview_path = str(selected_path)

                    import numpy as _np
                    if pixel_representation == "Black-white (1 bit/pixel)":
                        arr = (_np.asarray(im, dtype=_np.uint8) > 0).astype(_np.uint8).ravel()
                        raw = _np.packbits(arr).tobytes()
                    else:
                        raw = im.tobytes()
                    bit_text = _bytes_to_bit_text(raw)
                except Exception as exc:
                    preview_error = str(exc)

            preview_left, preview_right = st.columns(2, gap="small")
            with preview_left:
                preview_file(path, FIELDS["input_preview"])
            with preview_right:
                if selected_preview_path:
                    preview_file(selected_preview_path, "Selected pixel image")
                elif preview_error:
                    st.warning(f"Could not show selected pixel preview: {preview_error}")
                else:
                    st.info("Selected pixel preview is not available.")

            if bit_text:
                with st.expander("Raw pixel binary", expanded=False):
                    st.text_area(
                        FIELDS["binary_bitstream"],
                        bit_text[:3000] + ("..." if len(bit_text) > 3000 else ""),
                        height=120,
                    )
                    _download_text_button("Download raw pixel binary", bit_text, "raw_pixel_binary.txt")


# -----------------------------------------------------------------------------
# Panel 2 — Encode: compression
# -----------------------------------------------------------------------------







def render_panel_2_compression() -> None:
    with st.container(border=True):
        step_header(2, PANEL_TITLES["data_encoding"])
        data = st.session_state.get("input_bytes")
        path = st.session_state.get("input_path")
        if not data or not path:
            st.info(MESSAGES["upload_first"])
            return

        if not _is_uploaded_image(path, data):
            st.error("This pipeline supports image files only.")
            return

        pixel_representation = st.session_state.get("pixel_representation", "Grayscale (8 bits/pixel)")
        threshold = int(st.session_state.get("bw_threshold", 128))

        st.markdown("#### Image Compression")
        storage_mode = st.radio(
            "Storage mode",
            ["No compression", "Compression"],
            horizontal=True,
            index=0 if st.session_state.get("robust_image_storage_mode", "No compression") == "No compression" else 1,
            key="robust_image_storage_mode",
        )
        compression_method_options = [m for m in list_image_compression_methods() if m != "No compression"]
        previous_method = st.session_state.get("robust_image_compression_method", compression_method_options[0] if compression_method_options else "No compression")
        if storage_mode == "No compression":
            method = "No compression"
        else:
            if previous_method not in compression_method_options:
                previous_method = compression_method_options[0]
            method = st.selectbox(
                "Image compressor",
                compression_method_options,
                index=compression_method_options.index(previous_method),
                key="robust_image_compression_method",
            )

        compression_level = "No compression" if method == "No compression" else "Balanced"
        custom_params: Dict[str, Any] = {}

        if method != "No compression":
            compression_level = st.selectbox(
                "Compression level",
                list_compression_levels(),
                index=1,
                key="robust_image_compression_level",
            )

            if method == "Robust Low-Resolution":
                defaults = {
                    "High quality": {"downsample": 3, "bits_per_channel": 6},
                    "Balanced": {"downsample": 4, "bits_per_channel": 5},
                    "High compression": {"downsample": 6, "bits_per_channel": 4},
                    "Custom": {"downsample": 4, "bits_per_channel": 5},
                }[compression_level]

                with st.expander("Advanced settings", expanded=(compression_level == "Custom")):
                    custom_params["downsample"] = st.slider(
                        "Downsample",
                        min_value=2,
                        max_value=12,
                        value=int(defaults["downsample"]),
                        step=1,
                        key="lowres_downsample",
                    )
                    custom_params["bits_per_channel"] = st.slider(
                        "Bits per channel",
                        min_value=2,
                        max_value=8,
                        value=int(defaults["bits_per_channel"]),
                        step=1,
                        key="lowres_bits",
                    )
                if compression_level != "Custom":
                    custom_params = {}

            elif method == "Base + Local Detail":
                defaults = {
                    "High quality": {"base_downsample": 4, "base_bits": 6, "keep_coeffs": 6, "coeff_bits": 8, "q_step": 6.0},
                    "Balanced": {"base_downsample": 4, "base_bits": 5, "keep_coeffs": 4, "coeff_bits": 8, "q_step": 8.0},
                    "High compression": {"base_downsample": 6, "base_bits": 5, "keep_coeffs": 3, "coeff_bits": 8, "q_step": 10.0},
                    "Custom": {"base_downsample": 4, "base_bits": 5, "keep_coeffs": 4, "coeff_bits": 8, "q_step": 8.0},
                }[compression_level]

                with st.expander("Advanced settings", expanded=(compression_level == "Custom")):
                    custom_params["base_downsample"] = st.slider(
                        "Base downsample",
                        min_value=2,
                        max_value=12,
                        value=int(defaults["base_downsample"]),
                        step=1,
                        key="smart_base_downsample",
                    )
                    custom_params["base_bits"] = st.slider(
                        "Base bits",
                        min_value=2,
                        max_value=8,
                        value=int(defaults["base_bits"]),
                        step=1,
                        key="smart_base_bits",
                    )
                    custom_params["keep_coeffs"] = st.slider(
                        "Keep coefficients",
                        min_value=1,
                        max_value=12,
                        value=int(defaults["keep_coeffs"]),
                        step=1,
                        key="smart_keep_coeffs",
                    )
                    custom_params["coeff_bits"] = st.slider(
                        "Coefficient bits",
                        min_value=4,
                        max_value=10,
                        value=int(defaults["coeff_bits"]),
                        step=1,
                        key="smart_coeff_bits",
                    )
                    custom_params["q_step"] = st.number_input(
                        "Residual q step",
                        min_value=2.0,
                        max_value=30.0,
                        value=float(defaults["q_step"]),
                        step=0.5,
                        key="smart_q_step",
                    )
                if compression_level != "Custom":
                    custom_params = {}

            elif method == "Base + WebP Detail":
                defaults = {
                    "High quality": {"quality": 60, "tile_size": 128, "base_downsample": 4, "base_bits": 5},
                    "Balanced": {"quality": 45, "tile_size": 128, "base_downsample": 4, "base_bits": 5},
                    "High compression": {"quality": 30, "tile_size": 192, "base_downsample": 6, "base_bits": 4},
                    "Custom": {"quality": 45, "tile_size": 128, "base_downsample": 4, "base_bits": 5},
                }[compression_level]

                with st.expander("Advanced settings", expanded=(compression_level == "Custom")):
                    custom_params["quality"] = st.slider("WebP quality", 5, 95, int(defaults["quality"]), 1, key="base_webp_quality")
                    tile_opts = [64, 96, 128, 160, 192, 256]
                    custom_params["tile_size"] = st.selectbox("Tile size", tile_opts, index=tile_opts.index(int(defaults["tile_size"])), key="base_webp_tile")
                    custom_params["base_downsample"] = st.slider("Base downsample", 2, 10, int(defaults["base_downsample"]), 1, key="base_webp_base_downsample")
                    custom_params["base_bits"] = st.slider("Base bits", 2, 8, int(defaults["base_bits"]), 1, key="base_webp_base_bits")
                if compression_level != "Custom":
                    custom_params = {}

            elif method == "Local Block Coding":
                defaults = {
                    "High quality": {"y_q_scale": 1.2, "c_q_scale": 2.0, "y_packet_bits": 80, "c_packet_bits": 32},
                    "Balanced": {"y_q_scale": 1.5, "c_q_scale": 2.5, "y_packet_bits": 64, "c_packet_bits": 24},
                    "High compression": {"y_q_scale": 2.0, "c_q_scale": 3.5, "y_packet_bits": 48, "c_packet_bits": 16},
                    "Custom": {"y_q_scale": 1.5, "c_q_scale": 2.5, "y_packet_bits": 64, "c_packet_bits": 24},
                }[compression_level]

                with st.expander("Advanced settings", expanded=(compression_level == "Custom")):
                    custom_params["y_q_scale"] = st.number_input("Y q scale", min_value=0.5, max_value=5.0, value=float(defaults["y_q_scale"]), step=0.1, key="block_y_q")
                    custom_params["c_q_scale"] = st.number_input("C q scale", min_value=0.5, max_value=8.0, value=float(defaults["c_q_scale"]), step=0.1, key="block_c_q")
                    y_opts = [40, 48, 56, 64, 72, 80, 96]
                    c_opts = [16, 24, 32, 40, 48]
                    custom_params["y_packet_bits"] = st.selectbox("Y packet bits", y_opts, index=y_opts.index(int(defaults["y_packet_bits"])), key="block_y_bits")
                    custom_params["c_packet_bits"] = st.selectbox("C packet bits", c_opts, index=c_opts.index(int(defaults["c_packet_bits"])), key="block_c_bits")
                if compression_level != "Custom":
                    custom_params = {}
        if st.button(BUTTONS["run_data_encoding"], key="run_robust_image_compression"):
            _clear_downstream_from_storage()
            try:
                out_dir = WORK_ROOT / "robust_image_payload"
                payload, image_meta, preview_path = encode_image_to_payload(
                    path,
                    method,
                    out_dir=out_dir,
                    compression_level=compression_level,
                    custom_params=custom_params,
                    pixel_representation=pixel_representation,
                    threshold=threshold,
                )
                payload_path = out_dir / f"{image_meta['method']}_payload.bin"
                payload_path.write_bytes(payload)

                storage_meta = {
                    "kind": "robust_image_payload",
                    "method": method,
                    "compression_level": image_meta.get("compression_level", compression_level),
                    "pixel_representation": pixel_representation,
                    "image_codec_meta": image_meta,
                    "preview_path": preview_path,
                    "selected_pixel_preview_path": image_meta.get("selected_pixel_preview_path", preview_path),
                    "source": "image_content_payload" if method != "No compression" else "raw_pixel_payload",
                }

                st.session_state.update({
                    "compression_candidates": [],
                    "selected_candidate": None,
                    "stored_bytes": payload,
                    "stored_file_path": str(payload_path),
                    "storage_method": method,
                    "storage_kind": "robust_image_payload",
                    "storage_meta": storage_meta,
                })
            except Exception as exc:
                st.error(f"Image compression failed: {exc}")
                return

        stored = st.session_state.get("stored_bytes")
        if not stored:
            st.info("Run Image Compression first.")
            return

        storage_meta = st.session_state.get("storage_meta", {}) or {}
        image_meta = storage_meta.get("image_codec_meta", {}) or storage_meta
        before_path = storage_meta.get("preview_path", "")
        selected_path = storage_meta.get("selected_pixel_preview_path", "")

        raw_size = int(image_meta.get("raw_pixel_bytes", 0))
        upload_size = len(data)
        payload_size = len(stored)
        ratio = raw_size / max(1, payload_size)
        saving_pct = (1.0 - payload_size / max(1, raw_size)) * 100.0 if raw_size else 0.0
        upload_saving_pct = (1.0 - payload_size / max(1, upload_size)) * 100.0 if upload_size else 0.0

        payload_label = "Payload data" if storage_meta.get("method") == "No compression" else "Compressed payload"
        ratio_label = "Payload ratio" if storage_meta.get("method") == "No compression" else "Compression ratio"

        s1, s2, s3 = st.columns(3)
        s1.metric("Uploaded file", fmt_bytes(upload_size))
        s2.metric("Raw pixel data", fmt_bytes(raw_size))
        s3.metric(payload_label, fmt_bytes(payload_size))

        c1, c2, c3 = st.columns(3)
        c1.metric(ratio_label, f"{ratio:.2f}×")
        c2.metric("Reduced vs uploaded file", f"{upload_saving_pct:.1f}%")
        c3.metric("Reduced vs raw pixels", f"{saving_pct:.1f}%")

        p1, p2 = st.columns(2)
        with p1:
            if selected_path:
                preview_file(selected_path, "Selected pixel image")
            else:
                preview_file(path, "Input image")
        with p2:
            if storage_meta.get("method") == "No compression":
                if selected_path:
                    preview_file(selected_path, "Payload preview")
                else:
                    st.info("Payload preview is not available.")
            else:
                if before_path:
                    preview_file(before_path, "Before DNA error")
                else:
                    st.info("Compressed preview is not available.")

        # Compression-stage image quality: selected pixel image vs before-error preview.
        compare_path = before_path if before_path else selected_path
        quality_rows = [
            {"Metric": "Image size", "Value": f"{image_meta.get('pixel_width', image_meta.get('width', 0))} × {image_meta.get('pixel_height', image_meta.get('height', 0))}"},
            {"Metric": "Pixel representation", "Value": image_meta.get("pixel_representation", pixel_representation)},
            {"Metric": "Uploaded file", "Value": fmt_bytes(upload_size)},
            {"Metric": "Raw pixel data", "Value": fmt_bytes(raw_size)},
            {"Metric": payload_label, "Value": fmt_bytes(payload_size)},
            {"Metric": ratio_label, "Value": f"{ratio:.2f}×"},
            {"Metric": "Reduced vs uploaded file", "Value": f"{upload_saving_pct:.1f}%"},
            {"Metric": "Reduced vs raw pixels", "Value": f"{saving_pct:.1f}%"},
        ]
        if selected_path and compare_path and Image is not None:
            try:
                q = image_metrics(selected_path, compare_path)
                if q.get("Validation"):
                    for k, v in q.items():
                        if k != "Validation":
                            quality_rows.append({"Metric": k, "Value": v})
            except Exception:
                pass
        st.markdown("#### Compression quality")
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            download_bytes_button("Download image payload", stored, file_name="image_payload.bin")
        with d2:
            _download_text_button(BUTTONS["download_stored_binary"], _bytes_to_bit_text(stored), DOWNLOAD_FILES["stored_binary"])


def render_panel_3_encoding() -> None:
    with st.container(border=True):
        step_header(3, PANEL_TITLES["dna_encoding"])
        payload = st.session_state.get("stored_bytes")
        if not payload:
            st.info(MESSAGES["run_data_encoding_first"])
            return

        previous = st.session_state.get("encoding_mapping", MAPPING_OPTIONS[0] if MAPPING_OPTIONS else "Simple Mapping")
        if previous not in MAPPING_OPTIONS:
            previous = MAPPING_OPTIONS[0] if MAPPING_OPTIONS else "Simple Mapping"
        mapping = st.selectbox(
            FIELDS["dna_mapping"],
            MAPPING_OPTIONS,
            index=MAPPING_OPTIONS.index(previous),
            format_func=_display_mapping,
            key="encoding_mapping_select",
        )

        if st.button(BUTTONS["run_dna_encoding"], key="run_encoding"):
            dna, bits, meta = encode_bytes_to_dna(payload, mapping)
            st.session_state.update({
                "encoding_mapping": mapping,
                "dna": dna,
                "bits": bits,
                "codec_meta": meta,
                "strand_rows": [],
                "advanced_error_rows": [],
                "reads": [],
                "wetlab_metrics": {},
                "reconstructed_rows": [],
                "reconstructed_dna": "",
                "reconstruction_metrics": {},
                "decoded_data": None,
                "restored_info": None,
                "decode_error": "",
            })

        dna = st.session_state.get("dna", "")
        meta = st.session_state.get("codec_meta", {}) or {}
        if not dna:
            return

        baseline_nt = max(1, len(payload) * 4)
        dna_expansion = len(dna) / baseline_nt
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(METRICS["dna_mapping"], _display_mapping(st.session_state.get("encoding_mapping", mapping)))
        c2.metric(METRICS["dna_length"], f"{len(dna):,} nt")
        c3.metric(METRICS.get("dna_expansion", "DNA expansion"), f"{dna_expansion:.2f}×")
        c4.metric(METRICS["gc_content"], f"{gc_content(dna):.3f}")
        c5.metric(METRICS["longest_hp"], homopolymer_stats(dna).get("longest", 0))

        st.text_area(FIELDS["base_string"], _preview_seq(dna, 600), height=DNA_PREVIEW_HEIGHT)
        _download_text_button(BUTTONS["download_encoded_dna"], dna, DOWNLOAD_FILES["encoded_dna"])
        _download_text_button(BUTTONS["download_encoded_binary"], st.session_state.get("bits", ""), DOWNLOAD_FILES["encoded_binary"])

        # if meta.get("mode") == "NEW_DESIGN":
        #     st.markdown("#### Design Method A repair")
        #     st.dataframe(pd.DataFrame([{
        #         "Data block": meta.get("block_data_len"),
        #         "CRC8 nt/block": meta.get("crc8_symbols"),
        #         "State nt/block": meta.get("state_symbols"),
        #         "Blocks": meta.get("num_blocks"),
        #         "Repair added": meta.get("overhead_symbols"),
        #         "Expansion": meta.get("expansion_factor"),
        #     }]), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Panel 4 — DNA Strand Prep + Advanced errors + Wet-lab
# -----------------------------------------------------------------------------




def render_panel_4_experiment() -> None:
    with st.container(border=True):
        step_header(4, PANEL_TITLES["strand_preparation"])
        dna = st.session_state.get("dna", "")
        if not dna:
            st.info(MESSAGES["run_dna_encoding_first"])
            return

        mapping = st.session_state.get("encoding_mapping", "")
        codec_meta = st.session_state.get("codec_meta", {}) or {}
        is_rs = mapping == "Reed-Solomon" or codec_meta.get("mode") == "REED_SOLOMON"

        # ------------------------------------------------------------------
        # A. Strand Design
        # ------------------------------------------------------------------
        st.markdown(f"#### {PANEL_TITLES['strand_preparation']}")
        if is_rs:
            if st.button(BUTTONS["run_strand_preparation"], key="build_rs_strands"):
                rows = list(codec_meta.get("rs_strands", []) or [])
                if not rows:
                    st.error("Reed-Solomon strand metadata is missing. Run DNA Encoding again.")
                else:
                    st.session_state.update({
                        "strand_rows": rows,
                        "advanced_error_rows": [],
                        "noisy_dna": "",
                        "dna_error_stats": {},
                        "wetlab_input_source": "Designed strands",
                        "reads": [],
                        "wetlab_metrics": {},
                        "reconstructed_rows": [],
                        "reconstructed_dna": "",
                        "reconstruction_metrics": {},
                        "decoded_data": None,
                        "restored_info": None,
                    })
        else:
            with st.expander(FIELDS["strand_design"], expanded=not bool(st.session_state.get("strand_rows"))):
                target_len = st.number_input(FIELDS["total_strand_length"], min_value=80, max_value=250, value=125, step=1, key="std_total_len")
                index_len = st.number_input(FIELDS["si_length"], min_value=0, max_value=24, value=8, step=1, key="std_index_len")
                fbr = st.text_input(FIELDS["fbr"], value="ACACGACGCTCTTCCGATCT", key="std_fbr")
                rbr = st.text_input(FIELDS["rbr"], value="AGATCGGAAGAGCACACGTCT", key="std_rbr")
                if st.button(BUTTONS["run_strand_preparation"], key="build_standard_strands"):
                    cfg = choose_auto_strand_design(
                        len(dna), len(clean_dna(fbr)), len(clean_dna(rbr)), int(index_len),
                        min_total_len=int(target_len), max_total_len=int(target_len),
                    )
                    rows = prepare_dna_strands(
                        dna,
                        fbr=clean_dna(fbr),
                        rbr=clean_dna(rbr),
                        index_len=int(index_len),
                        target_total_len=int(cfg.get("target_total_len", cfg.get("total_len", target_len))),
                        add_filler=True,
                    )
                    for r in rows:
                        r["Type"] = FIELDS["prepared_strand"]
                    st.session_state.update({
                        "strand_rows": rows,
                        "advanced_error_rows": [],
                        "noisy_dna": "",
                        "dna_error_stats": {},
                        "wetlab_input_source": "Designed strands",
                        "reads": [],
                        "wetlab_metrics": {},
                        "reconstructed_rows": [],
                        "reconstructed_dna": "",
                        "reconstruction_metrics": {},
                        "decoded_data": None,
                        "restored_info": None,
                    })

        rows: List[Dict[str, Any]] = st.session_state.get("strand_rows", [])
        if not rows:
            st.info(MESSAGES["run_strand_preparation"])
            return

        total_strand_len = sum(len(clean_dna(r.get("Full strand", ""))) for r in rows)
        stored_len = len(st.session_state.get("stored_bytes", b"") or b"")
        strand_baseline_nt = max(1, stored_len * 4)
        strand_expansion = total_strand_len / strand_baseline_nt
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(METRICS["prepared_strands"], len(rows))
        c2.metric(METRICS["total_strand_length"], f"{total_strand_len:,} nt")
        c3.metric(METRICS.get("strand_design_expansion", "Strand Design expansion"), f"{strand_expansion:.2f}×")
        c4.metric(METRICS["dna_mapping"], _display_mapping(mapping or "—"))

        st.dataframe(_strand_summary(rows), use_container_width=True, hide_index=True)
        inspect_ids = [str(r.get("No.", i + 1)) for i, r in enumerate(rows)]
        selected_no = st.selectbox(FIELDS["inspect_prepared_strand"], inspect_ids, index=0, key="inspect_prepared_strand")
        selected_row = next((r for r in rows if str(r.get("No.", "")) == selected_no), rows[0])
        _render_segmented_strand(selected_row, FIELDS["prepared_strand"])
        st.download_button(BUTTONS["download_prepared_strands"], data=strand_rows_to_csv(rows), file_name=DOWNLOAD_FILES["prepared_strands"], mime="text/csv", use_container_width=True)

        # ------------------------------------------------------------------
        # B. DNA Error Simulation at prepared-strand level
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### DNA Error Simulation")
        enable_strand_errors = st.checkbox(
            FIELDS.get("add_strand_errors", "Add DNA errors"),
            value=bool(st.session_state.get("advanced_error_rows")),
            key="enable_strand_level_errors",
        )

        if enable_strand_errors:
            with st.container(border=True):
                a, b, c, d = st.columns(4)
                adv_scope = a.selectbox("Error target", ["Payload only", "Index + Payload", "Full strand"], index=0, key="adv_scope")
                adv_sub = b.number_input(FIELDS["substitution"], min_value=0.0, max_value=0.2, value=0.002, step=0.001, format="%.4f", key="adv_sub")
                adv_ins = c.number_input(FIELDS["insertion"], min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="adv_ins")
                adv_del = d.number_input(FIELDS["deletion"], min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key="adv_del")
                adv_seed = 17
                adv_indels = bool(float(adv_ins) > 0 or float(adv_del) > 0)

                if st.button(BUTTONS["run_add_errors"], key="run_advanced_errors"):
                    err_rows = []
                    for row in rows:
                        row_no = int(str(row.get("No.", "0") or "0"))
                        err_rows.append(_mutate_prepared_strand(
                            row,
                            scope=adv_scope,
                            substitution_rate=float(adv_sub),
                            insertion_rate=float(adv_ins),
                            deletion_rate=float(adv_del),
                            seed=int(adv_seed) + row_no * 1000003,
                            allow_indels=bool(adv_indels),
                        ))

                    noisy_dna = _dna_from_strand_rows_for_decode(
                        err_rows,
                        mapping,
                        original_dna_len=len(clean_dna(st.session_state.get("dna", ""))),
                    )
                    events = _advanced_error_events_table(err_rows)
                    st.session_state.update({
                        "advanced_error_rows": err_rows,
                        "noisy_dna": noisy_dna,
                        "dna_error_stats": {
                            "error_target": adv_scope,
                            "substitution_rate": float(adv_sub),
                            "insertion_rate": float(adv_ins),
                            "deletion_rate": float(adv_del),
                            "allow_indels": bool(adv_indels),
                            "seed": int(adv_seed),
                            "total_errors": int(len(events)),
                            "noisy_dna_len": int(len(noisy_dna)),
                        },
                        "wetlab_input_source": "Error strands",
                        "reads": [],
                        "wetlab_metrics": {},
                        "reconstructed_rows": [],
                        "reconstructed_dna": "",
                        "reconstruction_metrics": {},
                        "decoded_data": None,
                        "restored_info": None,
                    })
        else:
            if st.session_state.get("wetlab_input_source") == "Error strands":
                st.session_state["wetlab_input_source"] = "Designed strands"
            if st.session_state.get("advanced_error_rows") or st.session_state.get("noisy_dna"):
                st.session_state["advanced_error_rows"] = []
                st.session_state["noisy_dna"] = ""
                st.session_state["dna_error_stats"] = {}

        err_rows: List[Dict[str, Any]] = st.session_state.get("advanced_error_rows", []) if enable_strand_errors else []
        if err_rows:
            events = _advanced_error_events_table(err_rows)
            m1, m2, m3 = st.columns(3)
            m1.metric(METRICS["error_strands"], len(err_rows))
            m2.metric(METRICS["added_errors"], len(events))
            m3.metric("Noisy encoded data", f"{len(st.session_state.get('noisy_dna', '')):,} nt")
            st.dataframe(_advanced_error_rows_table(err_rows), use_container_width=True, hide_index=True)

            eids = [str(r.get("No.", i + 1)) for i, r in enumerate(err_rows)]
            eno = st.selectbox(FIELDS["inspect_error_strand"], eids, index=0, key="inspect_error_strand")
            erow = next((r for r in err_rows if str(r.get("No.", "")) == eno), err_rows[0])
            clean_for_error = next((r for r in rows if str(r.get("No.", "")) == eno), rows[0])
            try:
                ev_list = json.loads(erow.get("Advanced error events", "[]") or "[]")
            except Exception:
                ev_list = []
            err_positions = {int(ev.get("position_original")) for ev in ev_list if ev.get("operation") in {"substitution", "deletion"} and str(ev.get("position_original", "")).isdigit()}
            _render_segmented_strand(clean_for_error, FIELDS["clean_strand"], error_positions=err_positions)

            display_error_row = dict(erow)
            if int(str(erow.get("Insertion count", "0") or "0")) == 0 and int(str(erow.get("Deletion count", "0") or "0")) == 0:
                full_err = clean_dna(erow.get("Error full strand", ""))
                cursor = 0
                for name, _seq in _row_regions(clean_for_error):
                    key = "Strand index" if name == "SI" else name
                    n = len(clean_dna(clean_for_error.get(key, "")))
                    display_error_row[key] = full_err[cursor:cursor + n]
                    cursor += n
            _render_segmented_strand(display_error_row, FIELDS["error_strand"], error_positions=err_positions)

            if not events.empty:
                st.dataframe(events, use_container_width=True, hide_index=True)
                st.download_button(BUTTONS["download_error_table"], data=_df_csv_bytes(events), file_name=DOWNLOAD_FILES["error_table"], mime="text/csv", use_container_width=True)
            st.download_button(BUTTONS["download_error_strands"], data=strand_rows_to_csv(err_rows), file_name=DOWNLOAD_FILES["error_strands"], mime="text/csv", use_container_width=True)
            _download_text_button("Download noisy encoded DNA", st.session_state.get("noisy_dna", ""), "noisy_encoded_dna.txt")

        # ------------------------------------------------------------------
        # C. Advanced Sequencing Simulation
        # ------------------------------------------------------------------
        with st.expander("Advanced sequencing simulation", expanded=False):
            input_options = ["Designed strands"]
            if err_rows:
                input_options.append("Error strands")
            default_idx = 1 if st.session_state.get("wetlab_input_source") == "Error strands" and err_rows else 0
            wetlab_input_source = st.radio(FIELDS["sequencing_input"], input_options, index=default_idx, horizontal=True, key="wetlab_input_choice")
            st.session_state["wetlab_input_source"] = wetlab_input_source
            wetlab_rows = err_rows if wetlab_input_source == "Error strands" and err_rows else rows

            a, b, c, d, e = st.columns(5)
            coverage = a.number_input(FIELDS["coverage"], min_value=1, max_value=100, value=10, step=1, key="err_cov")
            sub = b.number_input(FIELDS["substitution"], min_value=0.0, max_value=0.2, value=0.001, step=0.001, format="%.4f", key="err_sub")
            ins = c.number_input(FIELDS["insertion"], min_value=0.0, max_value=0.2, value=0.0, step=0.001, format="%.4f", key="err_ins")
            dele = d.number_input(FIELDS["deletion"], min_value=0.0, max_value=0.2, value=0.0, step=0.001, format="%.4f", key="err_del")
            dropout = e.number_input(FIELDS["dropout"], min_value=0.0, max_value=0.9, value=0.0, step=0.01, format="%.3f", key="err_dropout")
            seq_seed = 7

            if st.button(BUTTONS["run_sequencing_simulation"], key="run_wetlab_reads"):
                reads, wet_metrics = simulate_sequencing_reads(
                    wetlab_rows,
                    coverage=int(coverage),
                    substitution_rate=float(sub),
                    insertion_rate=float(ins),
                    deletion_rate=float(dele),
                    dropout_rate=float(dropout),
                    randomize_coverage=False,
                    seed=7,
                )
                st.session_state.update({
                    "reads": reads,
                    "wetlab_metrics": wet_metrics,
                    "reconstructed_rows": [],
                    "reconstructed_dna": "",
                    "reconstruction_metrics": {},
                    "decoded_data": None,
                    "restored_info": None,
                })

            reads = st.session_state.get("reads", [])
            if reads:
                st.markdown(f"##### {FIELDS.get('sequencing_result', 'Sequencing result')}")
                wm = st.session_state.get("wetlab_metrics", {}) or {}
                m1, m2, m3 = st.columns(3)
                m1.metric(METRICS["input_type"], wetlab_input_source)
                m2.metric(METRICS["sequencing_reads"], wm.get("reads_generated", len(reads)))
                m3.metric(METRICS["sequencing_read_errors"], wm.get("total_errors", 0))

                reads_df = _reads_table(reads)
                read_events = _error_events_table(reads)
                if not read_events.empty:
                    st.dataframe(read_events, use_container_width=True, hide_index=True)
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(BUTTONS["download_noisy_reads"], data=_df_csv_bytes(reads_df), file_name=DOWNLOAD_FILES["noisy_reads"], mime="text/csv", use_container_width=True)
                with d2:
                    st.download_button(BUTTONS["download_read_errors"], data=_df_csv_bytes(read_events), file_name=DOWNLOAD_FILES["read_errors"], mime="text/csv", use_container_width=True)

                st.markdown(f"#### {PANEL_TITLES['read_recovery']}")
                if st.button(BUTTONS["run_read_recovery"], key="run_reconstruction"):
                    recovery_cluster_method = "oracle_source" if is_rs else "index_aware"
                    rec_rows, rec_dna, rec_metrics = reconstruct_consensus_from_reads(
                        wetlab_rows,
                        reads,
                        cluster_method=recovery_cluster_method,
                    )
                    original_encoded_dna = clean_dna(st.session_state.get("dna", ""))

                    if is_rs:
                        recovered_for_decode = clean_dna(rec_dna)
                        decode_acc = _dna_accuracy(recovered_for_decode, original_encoded_dna) if original_encoded_dna else 0.0
                        decode_mismatches = _dna_distance(recovered_for_decode[:len(original_encoded_dna)], original_encoded_dna) if original_encoded_dna else 0
                    else:
                        recovered_for_decode = clean_dna(rec_dna)[:len(original_encoded_dna)] if original_encoded_dna else clean_dna(rec_dna)
                        decode_acc = _dna_accuracy(recovered_for_decode, original_encoded_dna)
                        decode_mismatches = _dna_distance(recovered_for_decode, original_encoded_dna)

                    rec_metrics = dict(rec_metrics)
                    rec_metrics.update({
                        "decode_payload_mismatches": decode_mismatches,
                        "decode_payload_accuracy": decode_acc,
                        "decode_payload_len": len(recovered_for_decode),
                        "original_payload_len": len(original_encoded_dna),
                    })
                    st.session_state.update({
                        "reconstructed_rows": rec_rows,
                        "reconstructed_dna": recovered_for_decode,
                        "reconstruction_metrics": rec_metrics,
                    })

            rec_rows = st.session_state.get("reconstructed_rows", [])
            if rec_rows:
                wm = st.session_state.get("wetlab_metrics", {}) or {}
                rm = st.session_state.get("reconstruction_metrics", {}) or {}
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(METRICS["sequencing_reads"], wm.get("reads_generated", len(st.session_state.get("reads", []))))
                m2.metric(METRICS["recovered_strands"], rm.get("reconstructed_strands", 0))
                m3.metric(METRICS["reads_recovered"], f"{float(rm.get('payload_consensus_accuracy', 0)):.4f}")
                m4.metric(METRICS["dna_ready_for_decoding"], f"{float(rm.get('decode_payload_accuracy', 0)):.4f}")

                rec_df = _reconstruction_table(rec_rows)
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
                rec_dna = st.session_state.get("reconstructed_dna", "")
                st.text_area(FIELDS["recovered_dna"], _preview_seq(rec_dna, 900), height=DNA_PREVIEW_HEIGHT)
                d1, d2 = st.columns(2)
                with d1:
                    _download_text_button(BUTTONS["download_recovered_dna"], rec_dna, DOWNLOAD_FILES["recovered_dna"])
                with d2:
                    st.download_button(BUTTONS["download_recovery_table"], data=_df_csv_bytes(rec_df), file_name=DOWNLOAD_FILES["recovery_table"], mime="text/csv", use_container_width=True)


# -----------------------------------------------------------------------------
# Panel 5 — Decode
# -----------------------------------------------------------------------------



def render_panel_5_decoding() -> None:
    with st.container(border=True):
        step_header(5, PANEL_TITLES["file_decoding"])
        mapping = st.session_state.get("encoding_mapping")
        if not mapping:
            st.info(MESSAGES["run_dna_encoding_first"])
            return

        uploaded_strands = st.file_uploader(
            "Upload strands CSV for decode",
            type=["csv"],
            key="image_decode_upload_strands_csv",
        )
        if uploaded_strands is not None:
            try:
                upload_rows = _strand_rows_from_uploaded_csv(uploaded_strands)
                upload_dna = _dna_from_strand_rows_for_decode(
                    upload_rows,
                    mapping,
                    original_dna_len=len(clean_dna(st.session_state.get("dna", ""))),
                )
                st.session_state["uploaded_decode_strand_rows"] = upload_rows
                st.session_state["uploaded_decode_dna"] = upload_dna
                st.success(f"Loaded {len(upload_rows):,} uploaded strands for decode.")
            except Exception as exc:
                st.error(f"Could not load strands CSV: {exc}")

        source_label, dna_text = _decode_source()

        c1, c2 = st.columns(2)
        c1.metric(METRICS["dna_mapping"], _display_mapping(mapping))
        c2.metric("Reconstruction source", source_label)
        if not dna_text:
            st.info("No DNA is available for the selected reconstruction source.")
            return
        st.text_area(FIELDS["input_dna_preview"], _preview_seq(dna_text, 600), height=120)

        if st.button(BUTTONS["run_decode"], key="run_decode"):
            try:
                payload_data, bits, meta = decode_dna_with_mapping(
                    dna_text,
                    mapping,
                    codec_meta=st.session_state.get("codec_meta", {}) or {},
                )
                storage_meta = st.session_state.get("storage_meta", {}) or {}
                decoded_output = payload_data
                decoded_raw_pixels = None
                raw_restore_info: Dict[str, Any] = {}
                image_reconstruction_stats: Dict[str, Any] = {}

                if storage_meta.get("kind") == "robust_image_payload":
                    out_dir = WORK_ROOT / "robust_image_reconstruction"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "reconstructed_image.png"
                    image_meta = storage_meta.get("image_codec_meta", {}) or {}
                    image_reconstruction_stats = decode_payload_to_image(
                        payload_data,
                        image_meta,
                        str(out_path),
                    )
                    decoded_output = out_path.read_bytes()
                    m = detect_magic(decoded_output)
                    valid = True
                    note = "Image reconstructed from robust image payload."
                    info = _validate_and_write(decoded_output, preferred="reconstructed_image")
                    info["preview_path"] = str(out_path)
                    info["file_path"] = str(out_path)

                elif storage_meta.get("kind") == "raw_image_pixels":
                    decoded_raw_pixels = payload_data
                    decoded_output, raw_restore_info = _raw_image_bytes_to_png(payload_data, storage_meta)
                    m = detect_magic(decoded_output)
                    valid = True
                    note = f"Raw image pixels restored as PNG. {raw_restore_info.get('note', '')}"
                    info = _validate_and_write(decoded_output, preferred="restored_raw_image")

                else:
                    m = detect_magic(decoded_output)
                    valid = False
                    note = "No recognizable file signature"
                    if m:
                        valid, note = validate_container(decoded_output, m.kind)
                    info = _validate_and_write(decoded_output, preferred="restored")

                st.session_state.update({
                    "decoded_payload_bytes": payload_data,
                    "decoded_data": decoded_output,
                    "decoded_raw_pixels": decoded_raw_pixels,
                    "decoded_bits": bits,
                    "decoded_meta": meta,
                    "decoded_magic": m,
                    "decoded_valid": valid,
                    "decoded_note": note,
                    "raw_restore_info": raw_restore_info,
                    "image_reconstruction_stats": image_reconstruction_stats,
                    "restored_info": info,
                    "decode_source_label": source_label,
                    "reconstruction_source": source_label,
                    "decode_error": "",
                })
            except Exception as exc:
                st.session_state["decode_error"] = str(exc)
                st.session_state["restored_info"] = None

        if st.session_state.get("decode_error"):
            st.error(st.session_state["decode_error"])

        data = st.session_state.get("decoded_data")
        if data is None:
            return

        storage_meta = st.session_state.get("storage_meta", {}) or {}
        payload_bytes = st.session_state.get("decoded_payload_bytes", b"") or b""
        m = st.session_state.get("decoded_magic")
        valid = bool(st.session_state.get("decoded_valid"))

        c1, c2, c3 = st.columns(3)
        if storage_meta.get("kind") == "robust_image_payload":
            c1.metric("Decoded payload", fmt_bytes(len(payload_bytes)))
            c2.metric("Reconstructed type", m.kind if m else "unknown")
            c3.metric("Image can open", "Yes" if valid else "No")
        else:
            c1.metric(METRICS["decoded_size"], fmt_bytes(len(data)))
            c2.metric(METRICS["restored_type"], m.kind if m else "unknown")
            c3.metric(METRICS["file_can_open"], "Yes" if valid else "No")

        info = st.session_state.get("restored_info", {}) or {}
        preview_path = info.get("preview_path") or info.get("file_path")
        if preview_path:
            preview_file(preview_path, "Reconstructed preview")

        d1, d2 = st.columns(2)
        with d1:
            download_bytes_button(BUTTONS["download_decoded_file"], data, file_name=f"decoded{m.ext if m else '.bin'}")
        with d2:
            _download_text_button(BUTTONS["download_decoded_binary"], _bytes_to_bit_text(payload_bytes if payload_bytes else data), DOWNLOAD_FILES["decoded_binary"])

        stats = st.session_state.get("image_reconstruction_stats", {}) or {}
        if stats:
            st.markdown("#### Image reconstruction stats")
            st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)

        raw_pixels = st.session_state.get("decoded_raw_pixels")
        if raw_pixels is not None:
            r1, r2 = st.columns(2)
            with r1:
                download_bytes_button("Download decoded raw pixels", raw_pixels, file_name=DOWNLOAD_FILES["decoded_raw_pixels"])
            with r2:
                _download_text_button("Download decoded raw-pixel binary", _bytes_to_bit_text(raw_pixels), DOWNLOAD_FILES["decoded_raw_pixel_binary"])


# -----------------------------------------------------------------------------
# Panel 6 — Validate
# -----------------------------------------------------------------------------





def render_panel_6_analysis() -> None:
    with st.container(border=True):
        step_header(6, PANEL_TITLES["validation"])
        info = st.session_state.get("restored_info")
        if not info:
            st.info(MESSAGES["run_decode_first"])
            return

        storage_meta = st.session_state.get("storage_meta", {}) or {}
        image_meta = storage_meta.get("image_codec_meta", {}) or storage_meta
        stored_bytes = st.session_state.get("stored_bytes", b"") or b""

        before_path = storage_meta.get("preview_path") or st.session_state.get("input_path")
        after_path = info.get("preview_path") or info.get("file_path")

        decoded_payload = st.session_state.get("decoded_payload_bytes", b"") or b""
        if storage_meta.get("kind") == "raw_image_pixels":
            recovered_for_match = st.session_state.get("decoded_raw_pixels", b"") or decoded_payload
        elif storage_meta.get("kind") == "robust_image_payload":
            recovered_for_match = decoded_payload
        else:
            recovered_for_match = st.session_state.get("decoded_data", b"") or b""

        exact_payload_match = bool(stored_bytes) and bytes(recovered_for_match) == bytes(stored_bytes)

        st.markdown("#### Visual comparison")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("##### Before DNA error")
            if before_path:
                preview_file(before_path, "")
            else:
                st.info("Before-error image is not available.")

        with right:
            st.markdown("##### After DNA decoding")
            if after_path:
                preview_file(after_path, "")
            else:
                st.info("Decoded image is not available.")

        st.markdown("---")
        st.markdown("#### Comparison")

        raw_size = int(image_meta.get("raw_pixel_bytes", 0))
        img_w = image_meta.get("pixel_width", image_meta.get("width", 0))
        img_h = image_meta.get("pixel_height", image_meta.get("height", 0))
        image_size = f"{img_w} × {img_h}"
        pixel_depth = f"{image_meta.get('bits_per_pixel', '—')} bit/pixel"
        pixel_rep = image_meta.get("pixel_representation", storage_meta.get("pixel_representation", "—"))
        payload_size = len(stored_bytes)
        ratio = raw_size / max(1, payload_size) if raw_size else 0.0
        method_name = storage_meta.get("method", "—")
        payload_label = "Payload data" if method_name == "No compression" else "Compressed payload"
        ratio_label = "Payload ratio" if method_name == "No compression" else "Compression ratio"

        rows = [
            {"Metric": "Image size", "Value": image_size},
            {"Metric": "Pixel representation", "Value": pixel_rep},
            {"Metric": "Pixel depth", "Value": pixel_depth},
            {"Metric": "Compression method", "Value": method_name},
            {"Metric": "Compression level", "Value": image_meta.get("compression_level", "—")},
            {"Metric": "Raw pixel data", "Value": fmt_bytes(raw_size)},
            {"Metric": payload_label, "Value": fmt_bytes(payload_size)},
            {"Metric": ratio_label, "Value": f"{ratio:.2f}×"},
            {"Metric": "DNA design method", "Value": _display_mapping(st.session_state.get("encoding_mapping", "—"))},
            {"Metric": "Reconstruction source", "Value": st.session_state.get("decode_source_label", st.session_state.get("reconstruction_source", "—"))},
            {"Metric": "Payload exact match", "Value": "Yes" if exact_payload_match else "No"},
        ]

        dna_err = st.session_state.get("dna_error_stats", {}) or {}
        if dna_err and st.session_state.get("decode_source_label") == "Noisy encoded data":
            for key in ["error_target", "substitution_rate", "insertion_rate", "deletion_rate", "allow_indels", "total_errors", "noisy_dna_len"]:
                if key in dna_err:
                    rows.append({"Metric": key.replace("_", " ").title(), "Value": dna_err.get(key)})

        rec_stats = st.session_state.get("image_reconstruction_stats", {}) or {}
        if rec_stats:
            for key in ["valid_units", "invalid_units", "failure_rate", "decode_failed_units"]:
                if key in rec_stats:
                    val = rec_stats[key]
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    rows.append({"Metric": key.replace("_", " ").title(), "Value": val})

        # Image quality metrics: compare both against the before-error image and original upload.
        if before_path and after_path and Image is not None:
            try:
                m_before = image_metrics(before_path, after_path)
                if m_before.get("Validation"):
                    for k, v in m_before.items():
                        if k != "Validation":
                            rows.append({"Metric": f"Before-error vs decoded: {k}", "Value": v})
            except Exception:
                pass

        input_path = st.session_state.get("input_path")
        if input_path and after_path and Image is not None:
            try:
                m_orig = image_metrics(input_path, after_path)
                if m_orig.get("Validation"):
                    for k, v in m_orig.items():
                        if k != "Validation":
                            rows.append({"Metric": f"Original vs decoded: {k}", "Value": v})
            except Exception:
                pass

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if after_path:
            try:
                data = Path(after_path).read_bytes()
                download_bytes_button("Download reconstructed image", data, file_name="reconstructed_image.png", mime="image/png")
            except Exception:
                pass
