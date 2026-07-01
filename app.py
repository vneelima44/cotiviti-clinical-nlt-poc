import streamlit as st
import subprocess
import sys
import spacy
from openai import OpenAI
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical NLT Demo — Cotiviti Intern POC",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Clinical NLT Proof of Concept")
st.subheader("Named Entity Extraction + Automated ICD-10 Code Suggestion")
st.caption("Cotiviti Intern Assessment | Topic 1: Clinical Natural Language Technology for Health Care")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        help="Required for ICD-10 suggestions"
    )
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown(
        "- spaCy en_core_web_sm (NER)\n"
        "- OpenAI GPT-3.5-turbo (ICD coding)\n"
        "- Streamlit (UI)"
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "This POC demonstrates two core clinical NLT capabilities directly relevant to "
        "Cotiviti's Payment Accuracy and Risk Adjustment services: named entity recognition "
        "from clinical notes, and AI-assisted ICD-10 coding with evidence-grounded justifications."
    )
    st.markdown("---")
    st.markdown("**Relevant Cotiviti Services**")
    st.markdown(
        "- Coding Validation\n"
        "- Clinical Chart Validation\n"
        "- Medical Record Coding\n"
        "- Risk Adjustment"
    )

# ── Sample note ────────────────────────────────────────────────────────────────
SAMPLE_NOTE = (
    "Patient is a 67-year-old male presenting with a 3-day history of chest pain "
    "radiating to the left arm, shortness of breath, and diaphoresis. "
    "History of hypertension and type 2 diabetes mellitus. Currently on lisinopril 10mg "
    "daily and metformin 500mg twice daily. Troponin elevated at 2.4 ng/mL. "
    "EKG shows ST-segment elevation in leads II, III, and aVF. "
    "Assessment: Inferior ST-elevation myocardial infarction (STEMI). "
    "Plan: Emergent PCI, aspirin 325mg, heparin infusion, cardiology consult."
)

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Clinical Note Input")
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("📄 Load Sample Note", use_container_width=True):
        st.session_state["note"] = SAMPLE_NOTE

note = st.text_area(
    "Paste or type a de-identified clinical note:",
    value=st.session_state.get("note", ""),
    height=180,
    placeholder="e.g. 67-year-old male with chest pain, hypertension, type 2 diabetes...",
)

run = st.button("🔬 Analyze Note", type="primary", use_container_width=True)

# ── NLP model — download at runtime if not present ────────────────────────────
@st.cache_resource(show_spinner="Loading NLP model...")
def load_nlp_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True, capture_output=True
            )
            return spacy.load(model_name)
        except Exception:
            return None

def extract_entities(text):
    nlp = load_nlp_model()
    if nlp is None:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_, spacy.explain(ent.label_) or ent.label_) for ent in doc.ents]

# ── ICD suggestion ─────────────────────────────────────────────────────────────
def suggest_icd_codes(note_text, api_key):
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a certified medical coder (CPC). Given the following clinical note, "
        "identify the most relevant ICD-10-CM diagnosis codes for this encounter.\n\n"
        "For each code provide:\n"
        "1. The ICD-10-CM code (e.g. I21.19)\n"
        "2. The full code description\n"
        "3. A one-sentence clinical justification citing specific evidence from the note\n\n"
        "Format as a numbered list. Include principal diagnosis first, then secondary diagnoses.\n\n"
        f"Clinical Note:\n{note_text}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=700,
    )
    return response.choices[0].message.content

# ── Label color map ────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "PERSON": "#c0392b", "ORG": "#8e44ad", "GPE": "#2980b9",
    "DATE": "#27ae60", "TIME": "#16a085", "CARDINAL": "#d35400",
    "QUANTITY": "#f39c12", "NORP": "#1abc9c", "FAC": "#2c3e50",
    "LOC": "#2ecc71", "PRODUCT": "#e67e22", "EVENT": "#e74c3c",
    "WORK_OF_ART": "#9b59b6", "LAW": "#34495e", "LANGUAGE": "#1abc9c",
    "PERCENT": "#f1c40f", "MONEY": "#2ecc71", "ORDINAL": "#95a5a6",
}

def get_color(label):
    return LABEL_COLORS.get(label, "#1f4e79")

# ── Main output ────────────────────────────────────────────────────────────────
if run and note.strip():
    tab1, tab2 = st.tabs(
        ["🔍 Named Entity Recognition (spaCy)", "🏷️ ICD-10 Code Suggestions (LLM)"]
    )

    with tab1:
        st.markdown("### Clinical Named Entity Recognition")
        st.caption(
            "Using spaCy en_core_web_sm to identify and label named entities in clinical text. "
            "In production, a biomedical model (SciSpaCy en_core_sci_lg) would be used for "
            "richer clinical entity extraction including diseases, drugs, and procedures."
        )
        with st.spinner("Extracting entities..."):
            entities = extract_entities(note)

        if entities:
            st.markdown("**Identified Entities:**")
            tag_html = ""
            seen = set()
            for ent_text, label, explanation in entities:
                key = (ent_text.lower(), label)
                if key in seen:
                    continue
                seen.add(key)
                color = get_color(label)
                tag_html += (
                    f'<span style="background:{color};color:white;padding:4px 10px;'
                    f'border-radius:14px;margin:3px;display:inline-block;font-size:13px">'
                    f'<b>{ent_text}</b>'
                    f'<span style="opacity:0.8;font-size:11px"> [{label}]</span></span>'
                )
            st.markdown(tag_html, unsafe_allow_html=True)
            st.markdown("")

            with st.expander("📊 View as Table"):
                df = pd.DataFrame(
                    [(e[0], e[1], e[2]) for e in entities],
                    columns=["Entity", "Label", "Description"]
                )
                st.dataframe(df, use_container_width=True)

            st.info(
                f"✅ Extracted **{len(entities)} entity mentions** from the clinical note. "
                "In a Cotiviti production pipeline, entities would be normalized against "
                "SNOMED CT, RxNorm, and LOINC ontologies for downstream payment accuracy analytics."
            )
        else:
            st.warning("No entities found. Try loading the sample note or entering a more detailed note.")

    with tab2:
        st.markdown("### AI-Assisted ICD-10-CM Coding")
        st.caption(
            "Simulating Cotiviti's Coding Validation workflow: an LLM suggests diagnosis codes "
            "with evidence-grounded justifications for human reviewer approval."
        )
        if not openai_key:
            st.warning("⚠️ Enter your OpenAI API key in the sidebar to enable live ICD-10 suggestions.")
            st.markdown("**Example output (what you would see with an API key):**")
            st.markdown(
                "**1. I21.19** — ST elevation (STEMI) myocardial infarction involving other "
                "coronary artery of inferior wall  \n"
                "*Justification: Note documents inferior STEMI with ST-segment elevation in "
                "leads II, III, and aVF.*\n\n"
                "**2. I10** — Essential (primary) hypertension  \n"
                "*Justification: Patient has documented history of hypertension.*\n\n"
                "**3. E11.9** — Type 2 diabetes mellitus without complications  \n"
                "*Justification: History of type 2 diabetes mellitus documented as active comorbidity.*\n\n"
                "**4. Z79.899** — Other long-term (current) drug therapy  \n"
                "*Justification: Patient is on lisinopril and metformin, indicating active chronic pharmacotherapy.*"
            )
        else:
            with st.spinner("Consulting LLM for ICD-10 codes... (5-10 seconds)"):
                try:
                    icd_result = suggest_icd_codes(note, openai_key)
                    st.markdown(icd_result)
                    st.markdown("---")
                    st.success(
                        "💡 **Cotiviti Application:** This LLM output feeds into a human-in-the-loop "
                        "coding review workflow where certified coders validate AI suggestions before "
                        "claim adjudication — combining LLM speed with expert accuracy to reduce "
                        "the \$12B in annual healthcare waste that Cotiviti targets."
                    )
                except Exception as e:
                    st.error(f"OpenAI API error: {str(e)}")

elif run:
    st.warning("⚠️ Please enter a clinical note above before analyzing.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:grey;font-size:12px'>"
    "Clinical NLT POC · Cotiviti Intern Assessment · "
    "Technologies: NLP · OCR · Computer Vision · LLM · LMM · "
    "⚠️ For demonstration only — not for clinical use"
    "</div>",
    unsafe_allow_html=True,
)
