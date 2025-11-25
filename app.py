import json
import streamlit as st
import re
from datasets import load_from_disk

# ---------- Helper: Parse SOAP note ----------
def parse_soap(note: str):
    """
    Extracts S, O, A, P sections from a SOAP note string.
    Returns a dict with keys {"S", "O", "A", "P"}.
    Missing sections return empty strings.
    """
    if not isinstance(note, str):
        return {"S": "", "O": "", "A": "", "P": ""}

    pattern = r"(S:|O:|A:|P:)"
    parts = re.split(pattern, note)

    result = {"S": "", "O": "", "A": "", "P": ""}

    for i in range(1, len(parts), 2):
        section = parts[i].replace(":", "")
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result[section] = content

    return result


# ---------- Load Data ----------
@st.cache_data
def load_notes():
    # Load model predictions
    with open("./led_predicitons_test_epoch_2.json", "r") as f:
        led_notes = json.load(f)

    with open("./openai_predictions.json", "r") as f:
        openai_notes = json.load(f)

    # Load HuggingFace dataset from disk
    dataset = load_from_disk("./combined_normalized_data")

    # Extract test split soap notes
    gt_notes = dataset["test"]["soap_note"]

    return led_notes, openai_notes, gt_notes


led_notes, openai_notes, gt_notes = load_notes()

num_notes = min(len(led_notes), len(openai_notes), len(gt_notes))

# ---------- UI ----------
st.title("SOAP Note Comparator (3-Way: LED vs OpenAI vs Ground Truth)")

st.sidebar.header("Select Note")
selected_idx = st.sidebar.selectbox(
    "Choose a SOAP note index",
    options=list(range(num_notes)),
    format_func=lambda x: f"Note {x}"
)

# Parse SOAP components
note_led = parse_soap(led_notes[selected_idx])
note_openai = parse_soap(openai_notes[selected_idx])
note_gt = parse_soap(gt_notes[selected_idx])

st.header(f"Comparison for Note {selected_idx}")

sections = ["S", "O", "A", "P"]
section_names = {
    "S": "Subjective",
    "O": "Objective",
    "A": "Assessment",
    "P": "Plan"
}

# ---------- Section-by-section comparison ----------
for sec in sections:
    st.subheader(f"{sec}: {section_names[sec]}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**LED Prediction**")
        st.text(note_led[sec])

    with col2:
        st.markdown("**OpenAI Prediction**")
        st.text(note_openai[sec])

    with col3:
        st.markdown("**Ground Truth**")
        st.text(note_gt[sec])
