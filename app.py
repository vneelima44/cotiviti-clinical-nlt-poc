import streamlit as st
import re
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
        "- Rule-based NER (regex + keyword matching)\n"
        "- OpenAI GPT-3.5-turbo (ICD-10 coding)\n"
        "- Streamlit (UI)\n"
        "- In production: SciSpaCy biomedical NLP model"
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

# ── Clinical entity dictionaries ──────────────────────────────────────────────
DIAGNOSES = [
    "hypertension", "diabetes mellitus", "type 2 diabetes", "myocardial infarction",
    "STEMI", "chest pain", "shortness of breath", "diaphoresis", "ST-elevation",
    "pneumonia", "atrial fibrillation", "heart failure", "stroke", "sepsis",
    "COVID-19", "pulmonary embolism", "acute kidney injury", "anemia",
]
MEDICATIONS = [
    "lisinopril", "metformin", "aspirin", "heparin", "warfarin", "insulin",
    "metoprolol", "atorvastatin", "amlodipine", "furosemide", "prednisone",
    "amoxicillin", "albuterol", "clopidogrel", "pantoprazole",
]
PROCEDURES = [
    "PCI", "EKG", "echocardiogram", "CT scan", "MRI", "X-ray", "biopsy",
    "catheterization", "intubation", "dialysis", "colonoscopy", "endoscopy",
]
LAB_VALUES = [
    "troponin", "hemoglobin", "creatinine", "glucose", "BNP", "INR", "TSH",
    "WBC", "potassium", "sodium", "lactate", "D-dimer", "HbA1c",
]

ENTITY_GROUPS = {
    "DIAGNOSIS": (DIAGNOSES, "#c0392b"),
    "MEDICATION": (MEDICATIONS, "#8e44ad"),
    "PROCEDURE": (PROCEDURES, "#2980b9"),
    "LAB VALUE": (LAB_VALUES, "#27ae60"),
}

# ── Age pattern ────────────────────────────────────────────────────────────────
AGE_PATTERN = re.compile(r"\\b(\\d{1,3})[- ]?year[s]?[- ]?old\\b", re.IGNORECASE)

def extract_clinical_entities(text):
    entities = []
    text_lower = text.lower()

    # Age
    for match in AGE_PATTERN.finditer(text):
        entities.append((match.group(0), "PATIENT AGE", "#e67e22"))

    # Keyword-based
    for label, (keywords, color) in ENTITY_GROUPS.items():
        for kw in keywords:
            pattern = re.compile(r"\\b" + re.escape(kw) + r"\\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append((match.group(0), label, color))

    # Dosage pattern (e.g. 10mg, 325mg)
    dose_pattern = re.compile(r"\\b\\d+(?:\\.\\d+)?\\s?(?:mg|mcg|mEq|units?|IU|mL|g)\\b", re.IGNORECASE)
    for match in dose_pattern.finditer(text):
        entities.append((match.group(0), "DOSAGE", "#16a085"))

    # Deduplicate by (text, label)
    seen = set()
    result = []
    for ent in entities:
        key = (ent[0].lower(), ent[1])
        if key not in seen:
            seen.add(key)
            result.append(ent)

    return result

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

# ── Main output ────────────────────────────────────────────────────────────────
if run and note.strip():
    tab1, tab2 = st.tabs(
        ["🔍 Named Entity Recognition", "🏷️ ICD-10 Code Suggestions (LLM)"]
    )

    with tab1:
        st.markdown("### Clinical Named Entity Extraction")
        st.caption(
            "Rule-based NER using clinical keyword dictionaries and regex patterns. "
            "In production, this would use a SciSpaCy biomedical NLP model (en_core_sci_lg) "
            "for deeper extraction including rare diseases, drug interactions, and anatomical terms."
        )
        with st.spinner("Extracting entities..."):
            entities = extract_clinical_entities(note)

        if entities:
            st.markdown("**Identified Clinical Entities:**")
            tag_html = ""
            for ent_text, label, color in entities:
                tag_html += (
                    f'<span style="background:{color};color:white;padding:4px 10px;'
                    f'border-radius:14px;margin:3px;display:inline-block;font-size:13px">'
                    f'<b>{ent_text}</b>'
                    f'<span style="opacity:0.8;font-size:11px"> [{label}]</span></span>'
                )
            st.markdown(tag_html, unsafe_allow_html=True)
            st.markdown("")

            with st.expander("📊 View as Table"):
                df = pd.DataFrame(entities, columns=["Entity", "Label", "Color"])
                st.dataframe(df[["Entity", "Label"]], use_container_width=True)

            st.info(
                f"✅ Extracted **{len(entities)} clinical entities**. "
                "In a Cotiviti production pipeline, entities would be normalized against "
                "SNOMED CT, RxNorm, and LOINC ontologies for downstream payment accuracy analytics."
            )
        else:
            st.warning("No clinical entities found. Try loading the sample note.")

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
