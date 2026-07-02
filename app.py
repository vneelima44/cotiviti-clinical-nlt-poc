import streamlit as st
import re
from openai import OpenAI
import pandas as pd

st.set_page_config(
    page_title="Clinical NLT Demo - Cotiviti Intern POC",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Clinical NLT Proof of Concept")
st.subheader("Named Entity Extraction + Automated ICD-10 Code Suggestion")
st.caption("Cotiviti Intern Assessment | Topic 1: Clinical Natural Language Technology for Health Care")

with st.sidebar:
    st.header("Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Required for ICD-10 suggestions")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("- Rule-based clinical NER (keyword + regex)")
    st.markdown("- OpenAI GPT-3.5-turbo (ICD-10 coding)")
    st.markdown("- Streamlit (UI)")
    st.markdown("---")
    st.markdown("**Cotiviti Services**")
    st.markdown("- Coding Validation")
    st.markdown("- Clinical Chart Validation")
    st.markdown("- Medical Record Coding")
    st.markdown("- Risk Adjustment")

SAMPLE_NOTE = (
    "Patient is a 67-year-old male presenting with a 3-day history of chest pain "
    "radiating to the left arm, shortness of breath, and diaphoresis. "
    "History of hypertension and type 2 diabetes mellitus. Currently on lisinopril 10mg "
    "daily and metformin 500mg twice daily. Troponin elevated at 2.4 ng/mL. "
    "EKG shows ST-segment elevation in leads II, III, and aVF. "
    "Assessment: Inferior ST-elevation myocardial infarction (STEMI). "
    "Plan: Emergent PCI, aspirin 325mg, heparin infusion, cardiology consult."
)

DIAGNOSES = [
    "hypertension", "diabetes mellitus", "type 2 diabetes", "myocardial infarction",
    "STEMI", "chest pain", "shortness of breath", "diaphoresis", "ST-elevation",
    "pneumonia", "atrial fibrillation", "heart failure", "stroke", "sepsis",
    "pulmonary embolism", "acute kidney injury", "anemia", "COVID-19",
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

def extract_clinical_entities(text):
    entities = []
    age_pat = re.compile(r"\b(\d{1,3})[- ]?year[s]?[- ]?old\b", re.IGNORECASE)
    for m in age_pat.finditer(text):
        entities.append((m.group(0), "PATIENT AGE", "#e67e22"))
    for label, (keywords, color) in ENTITY_GROUPS.items():
        for kw in keywords:
            pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for m in pat.finditer(text):
                entities.append((m.group(0), label, color))
    dose_pat = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|mEq|units?|IU|mL|g)\b", re.IGNORECASE)
    for m in dose_pat.finditer(text):
        entities.append((m.group(0), "DOSAGE", "#16a085"))
    seen = set()
    result = []
    for ent in entities:
        key = (ent[0].lower(), ent[1])
        if key not in seen:
            seen.add(key)
            result.append(ent)
    return result

def suggest_icd_codes(note_text, api_key):
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a certified medical coder (CPC). Given the following clinical note, "
        "identify the most relevant ICD-10-CM diagnosis codes.\n\n"
        "For each code provide:\n"
        "1. ICD-10-CM code (e.g. I21.19)\n"
        "2. Full code description\n"
        "3. One-sentence justification from the note\n\n"
        "Format as numbered list. Principal diagnosis first.\n\n"
        f"Clinical Note:\n{note_text}"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=700,
    )
    return resp.choices[0].message.content

st.markdown("### Clinical Note Input")
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Load Sample Note", use_container_width=True):
        st.session_state["note"] = SAMPLE_NOTE

note = st.text_area(
    "Paste a de-identified clinical note:",
    value=st.session_state.get("note", ""),
    height=180,
    placeholder="e.g. 67-year-old male with chest pain, hypertension, type 2 diabetes...",
)

run = st.button("Analyze Note", type="primary", use_container_width=True)

if run and note.strip():
    tab1, tab2 = st.tabs(["Named Entity Recognition", "ICD-10 Code Suggestions (LLM)"])

    with tab1:
        st.markdown("### Clinical Named Entity Extraction")
        st.caption("Rule-based NER using clinical keyword dictionaries and regex patterns.")
        entities = extract_clinical_entities(note)
        if entities:
            st.markdown("**Identified Clinical Entities:**")
            tag_html = ""
            for ent_text, label, color in entities:
                tag_html += (
                    f'<span style="background:{color};color:white;padding:4px 10px;'
                    f'border-radius:14px;margin:3px;display:inline-block;font-size:13px">'
                    f'<b>{ent_text}</b> <span style="opacity:0.8;font-size:11px">[{label}]</span></span>'
                )
            st.markdown(tag_html, unsafe_allow_html=True)
            st.markdown("")
            with st.expander("View as Table"):
                df = pd.DataFrame([(e[0], e[1]) for e in entities], columns=["Entity", "Label"])
                st.dataframe(df, use_container_width=True)
            st.info(
                f"Extracted **{len(entities)} clinical entities**. "
                "In a Cotiviti production pipeline, these would be mapped to SNOMED CT, "
                "RxNorm, and LOINC for downstream payment accuracy analytics."
            )
        else:
            st.warning("No entities found. Try loading the sample note.")

    with tab2:
        st.markdown("### AI-Assisted ICD-10-CM Coding")
        st.caption("Simulating Cotiviti Coding Validation: LLM suggests codes for human reviewer approval.")
        if not openai_key:
            st.warning("Enter your OpenAI API key in the sidebar for live ICD-10 suggestions.")
            st.markdown("**Example output:**")
            st.markdown(
                "**1. I21.19** - ST elevation myocardial infarction (inferior wall)  \n"
                "*ST-segment elevation in leads II, III, aVF confirms inferior STEMI.*\n\n"
                "**2. I10** - Essential hypertension  \n"
                "*Documented history of hypertension.*\n\n"
                "**3. E11.9** - Type 2 diabetes mellitus  \n"
                "*Active comorbidity documented.*"
            )
        else:
            with st.spinner("Consulting LLM for ICD-10 codes..."):
                try:
                    result = suggest_icd_codes(note, openai_key)
                    st.markdown(result)
                    st.success(
                        "**Cotiviti Application:** This output feeds into a human-in-the-loop "
                        "coding review workflow — combining LLM speed with expert accuracy to "
                        "reduce the $12B in annual healthcare waste Cotiviti targets."
                    )
                except Exception as e:
                    st.error(f"API error: {e}")
elif run:
    st.warning("Please enter a clinical note above.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:grey;font-size:12px'>"
    "Clinical NLT POC | Cotiviti Intern Assessment | "
    "NLP · OCR · Computer Vision · LLM · LMM | For demonstration only"
    "</div>",
    unsafe_allow_html=True,
)
