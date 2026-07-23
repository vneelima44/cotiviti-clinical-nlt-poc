# 🏥 Clinical NLT Proof of Concept
### Cotiviti Intern Assessment — Topic 1: Clinical Natural Language Technology for Health Care

---

## What This Demo Does

A two-feature clinical NLP web application demonstrating technologies directly relevant to Cotiviti's Payment Accuracy and Risk Adjustment business:

1. **Named Entity Recognition (NER):** Extracts biomedical concepts from clinical notes using SciSpaCy's biomedical NLP model trained on PubMed and clinical text
2. **Automated ICD-10 Coding:** Suggests ICD-10-CM diagnosis codes with evidence-grounded justifications using OpenAI GPT-3.5-turbo

---

## Clinical Relevance to Cotiviti

This POC directly mirrors Cotiviti's core services:
- **Coding Validation** — AI suggests ICD/CPT codes; human coders validate before adjudication
- **Clinical Chart Validation** — NLP extracts clinical evidence from notes to support or contest claims
- **Medical Record Coding** — Automates the first pass of retrospective chart review for Risk Adjustment
- Addresses Cotiviti's mission of reducing an estimated **$12 billion** in annual healthcare waste

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Biomedical NER | spaCy + SciSpaCy (en_core_sci_sm) |
| ICD-10 Coding | OpenAI GPT-3.5-turbo |
| Language | Python 3.10+ |
| Deployment | Render |

---

## Local Setup

```bash
git clone https://github.com/vneelima44/cotiviti-clinical-nlt-poc.git
cd cotiviti-clinical-nlt-poc
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
streamlit run app.py
```

---

## Usage

1. Click **Load Sample Note** to load a STEMI clinical case
2. Click **Analyze Note** to run NLP entity extraction
3. Enter your OpenAI API key in the sidebar for live ICD-10 code suggestions
4. Switch tabs to view NER results and ICD-10 coding output

---

## Author

**Neelima Verma** | Pace University | Cotiviti Intern Assessment | June 2026

> This application is for demonstration purposes only and is not intended for clinical use.
