from __future__ import annotations

import streamlit as st
from text_sparse_semantic_dna_streamlit import render_text_dna_storage_panel

st.set_page_config(page_title="Text DNA Storage", page_icon="🧬", layout="wide")
st.title("🧬 Text DNA Storage")
render_text_dna_storage_panel()
