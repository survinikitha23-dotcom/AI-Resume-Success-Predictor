import html
import json
import os
import re
import time

import fitz
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_scoring import (
    GENERIC_JOB_TERMS,
    _is_recognized_skill_fragment,
    _keyword_is_present,
    _recognized_job_skills,
    _skill_key,
    calculate_keyword_match,
)


load_dotenv()


def calculate_resume_match(resume_text, job_description):
    if not resume_text.strip() or not job_description.strip():
        raise ValueError("Resume text and job description are required.")

    documents = [resume_text.strip(), job_description.strip()]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    document_vectors = vectorizer.fit_transform(documents)
    similarity = float(cosine_similarity(document_vectors[0], document_vectors[1])[0][0])

    terms = vectorizer.get_feature_names_out()
    job_weights = document_vectors[1].toarray().ravel()
    resume_lower = documents[0].lower()
    ranked_terms = sorted(
        range(len(terms)),
        key=lambda index: (" " not in terms[index], -job_weights[index]),
    )
    matching_terms = []
    recognized_skills = _recognized_job_skills(job_description)
    for key, label in recognized_skills:
        if _keyword_is_present(label, resume_text):
            matching_terms.append(label)

    for index in ranked_terms:
        term = terms[index]
        if job_weights[index] == 0:
            continue
        term_words = term.split()
        if any(word in GENERIC_JOB_TERMS for word in term_words):
            continue
        if _is_recognized_skill_fragment(term, recognized_skills):
            continue
        if any(set(term_words).issubset(existing.split()) for existing in matching_terms):
            continue
        if _keyword_is_present(term, resume_lower):
            matching_terms.append(term)
        if len(matching_terms) == 12:
            break

    return round(similarity * 100, 1), matching_terms


def generate_local_resume_feedback(resume_text, job_description):
    keyword_match = calculate_keyword_match(resume_text, job_description)
    matching_terms = keyword_match["matched_skills"]
    missing_terms = keyword_match["missing_skills"]

    suggestions = []
    if not keyword_match["detected_skills"]:
        suggestions.append(
            "No relevant skills or keywords were detected in the job description. "
            "Add specific requirements to enable a keyword comparison."
        )
    if missing_terms:
        suggestions.append(
            "If accurate, include evidence for these job-description keywords: "
            + ", ".join(missing_terms)
            + "; otherwise, do not claim them."
        )
    if matching_terms:
        suggestions.append(
            "Strengthen the resume bullets that support "
            + ", ".join(matching_terms[:4])
            + " with your action, context, and a result you can verify."
        )
    suggestions.append(
        "Prioritize completed projects that demonstrate the matched job requirements; "
        "state your actual contribution and tools, and do not add projects you did not do."
    )
    suggestions.append(
        "Use exact job-description keywords in your summary or bullets only where "
        "your resume provides truthful evidence."
    )
    suggestions.append(
        "For relevant experience, revise bullets to show action, context, and a "
        "verified outcome; do not invent metrics or responsibilities."
    )

    return {
        "relevant_skills_found": matching_terms,
        "possibly_missing_skills": missing_terms,
        "detected_job_skills": keyword_match["detected_skills"],
        "keyword_match_percentage": keyword_match["percentage"],
        "improvement_suggestions": suggestions,
    }


def assess_ats_compatibility(resume_text):
    text = resume_text.strip()
    words = re.findall(r"\b[\w+#.-]+\b", text)
    passed = []
    warnings = []

    if len(text) >= 100 and len(words) >= 15:
        passed.append("Readable resume text was extracted.")
    else:
        warnings.append(
            "Only a small amount of text was extracted; use a text-based PDF "
            "and check that all pages are readable."
        )

    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE):
        passed.append("An email address was found.")
    else:
        warnings.append("Add a clear professional email address near the top.")

    phone_match = re.search(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", text)
    if phone_match and 10 <= len(re.sub(r"\D", "", phone_match.group())) <= 15:
        passed.append("A phone-number-like contact was found.")
    else:
        warnings.append("Add a clearly formatted phone number if you want employers to call.")

    headings = (
        "experience", "work experience", "education", "skills", "projects",
        "summary", "certifications",
    )
    found_headings = [
        heading for heading in headings
        if re.search(r"(?im)^\s*" + re.escape(heading) + r"\s*:?\s*$", text)
    ]
    if found_headings:
        passed.append("Common section heading(s) found: " + ", ".join(found_headings[:4]) + ".")
    else:
        warnings.append(
            "Use clear standard headings such as Experience, Education, and Skills."
        )

    if len(text.splitlines()) >= 4:
        passed.append("Extracted text contains line breaks for section separation.")
    else:
        warnings.append(
            "Check the PDF reading order; use a simple layout with clear section "
            "breaks and avoid relying on complex columns or graphics."
        )

    return {"passed": passed, "warnings": warnings}


def _safe_card_badges(items, badge_class, empty_text="None identified."):
    if not items:
        return f'<span class="card-empty">{html.escape(empty_text)}</span>'
    badges = "".join(
        f'<span class="skill-badge skill-badge--{badge_class}">'
        f"{html.escape(item)}</span>"
        for item in items
    )
    return f'<div class="badge-list">{badges}</div>'


def _safe_card_list(items, empty_text="No recommendations available."):
    if not items:
        return f'<p class="card-empty">{html.escape(empty_text)}</p>'
    return "<ul class=\"card-list\">" + "".join(
        f"<li>{html.escape(item)}</li>" for item in items
    ) + "</ul>"


def build_feature_card_content(keyword_feedback, feedback, ats_report, feedback_source):
    detected = list(dict.fromkeys(keyword_feedback["detected_job_skills"]))
    matching_keys = {
        _skill_key(term) for term in keyword_feedback["relevant_skills_found"]
    }
    matching = [
        term for term in detected if _skill_key(term) in matching_keys
    ]
    missing = [
        term for term in detected if _skill_key(term) not in matching_keys
    ]
    keyword_count = len(detected)
    score = keyword_feedback["keyword_match_percentage"]
    score_label = f"{score:g}%" if score is not None else "N/A"
    score_detail = (
        f"{len(matching)} of {keyword_count} identified job-description "
        "skills/keywords found."
        if keyword_count
        else "No relevant skills or keywords were detected in the job description; "
        "a keyword match percentage is unavailable."
    )
    skill_card = (
        '<div class="score-overview">'
        f'<div class="score-ring" style="--score: {score or 0}%;" role="img" '
        f'aria-label="Keyword match score: {score_label}">'
        f'<span class="score-ring-value">{score_label}</span></div>'
        '<div class="score-copy"><span class="card-eyebrow">KEYWORD MATCH</span>'
        f'<strong>{html.escape(score_detail)}</strong>'
        '<span class="card-note">This is a keyword comparison, not a probability '
        'of getting hired. Text similarity (TF-IDF cosine) is shown above.</span>'
        '</div></div>'
        '<div class="card-section"><strong class="card-section-title">'
        'Matched skills and keywords</strong>'
        f'{_safe_card_badges(matching, "matched")}</div>'
        '<div class="card-section"><strong class="card-section-title">'
        'Missing from resume text</strong>'
        f'{_safe_card_badges(missing, "missing", "No missing keywords identified.")}'
        '</div>'
    )

    if feedback_source == "gemini":
        source_label = "Gemini suggestions; keyword and ATS checks are local"
        suggestions = list(feedback["improvement_suggestions"])
        suggestions.extend(keyword_feedback["improvement_suggestions"])
        suggestions = list(dict.fromkeys(suggestions))
    else:
        source_label = "Local keyword-based feedback (Gemini unavailable)"
        suggestions = keyword_feedback["improvement_suggestions"]
    improvement_card = (
        f'<span class="source-badge">{html.escape(source_label)}</span>'
        f'{_safe_card_list(suggestions, "No additional suggestions were identified.")}'
    )

    ats_report = ats_report or {"passed": [], "warnings": []}
    ats_card = (
        '<div class="card-section"><strong class="card-section-title">'
        'Checks passed</strong>'
        f'{_safe_card_badges(ats_report["passed"], "passed", "No checks passed.")}'
        '</div><div class="card-section"><strong class="card-section-title">'
        'Warnings and fixes</strong>'
        f'{_safe_card_badges(ats_report["warnings"], "warning", "No warnings.")}'
        '</div><p class="card-note">Basic text checks only — not a real ATS '
        'simulation or an acceptance guarantee.</p>'
    )

    next_actions = []
    if missing:
        next_actions.append(
            "Review the gaps and add evidence only for skills or experience you "
            "actually have."
        )
    if matching:
        next_actions.append(
            "Prepare a concrete example for "
            + ", ".join(matching[:3])
            + " to discuss in an interview."
        )
    next_actions.extend(
        [
            "Prioritize a completed project that demonstrates a target requirement "
            "and describe your actual contribution.",
            "Tailor the resume bullets to the job description using truthful "
            "keywords and verified outcomes.",
        ]
    )
    readiness_card = (
        '<div class="card-section"><strong class="card-section-title">'
        'Strengths</strong>'
        f'{_safe_card_badges(matching, "strength")}</div>'
        '<div class="card-section"><strong class="card-section-title">'
        'Important gaps</strong>'
        f'{_safe_card_badges(missing, "gap", "No specific gaps identified.")}</div>'
        '<div class="card-section"><strong class="card-section-title">'
        'Next actions</strong>'
        f'{_safe_card_list(next_actions[:3])}</div>'
        '<p class="card-note">Guidance only — not a guarantee of employment.</p>'
    )

    return {
        "skill_match": skill_card,
        "improvement_suggestions": improvement_card,
        "ats_compatibility": ats_card,
        "job_readiness": readiness_card,
    }


def _toggle_result_card(card_id):
    if st.session_state.active_result_card == card_id:
        st.session_state.active_result_card = None
    else:
        st.session_state.active_result_card = card_id


def _safe_gemini_error_diagnostic(error):
    exception_type = type(error)
    exception_class = exception_type.__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", exception_class):
        exception_class = "UnknownException"

    status_code = getattr(error, "code", None)
    if not isinstance(status_code, int) or isinstance(status_code, bool):
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    if not isinstance(status_code, int) or isinstance(status_code, bool):
        status_code = None

    if status_code == 429:
        reason_codes = set()

        def collect_reason_codes(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"reason", "status"} and isinstance(item, str):
                        normalized_code = (
                            item.strip().lower().replace("_", "").replace("-", "")
                        )
                        reason_codes.add(normalized_code)
                    elif key in {"error", "details", "errors"}:
                        collect_reason_codes(item)
            elif isinstance(value, list):
                for item in value:
                    collect_reason_codes(item)

        collect_reason_codes(getattr(error, "details", None))
        collect_reason_codes({"status": getattr(error, "status", None)})

        code_categories = {
            "ratelimit": "rate limit indicated",
            "ratelimitexceeded": "rate limit indicated",
            "userratelimitexceeded": "rate limit indicated",
            "quotaexceeded": "quota limit indicated",
            "quotaexhausted": "quota limit indicated",
            "dailylimitexceeded": "quota limit indicated",
            "billingdisabled": "billing restriction indicated",
            "billingnotenabled": "billing restriction indicated",
        }
        categories = {
            code_categories[code]
            for code in reason_codes
            if code in code_categories
        }
        category = (
            categories.pop()
            if len(categories) == 1
            else "429 cause unknown"
        )
    elif isinstance(error, json.JSONDecodeError):
        category = "JSON parsing error"
    elif exception_type.__module__ == "google.genai.errors":
        category = "Gemini API request error"
    elif isinstance(error, ValueError):
        category = "Response validation or processing error"
    else:
        category = "Other Gemini feedback error"

    status_text = str(status_code) if status_code is not None else "unavailable"
    return (
        f"Gemini diagnostic: {category}; "
        f"exception={exception_class}; HTTP status={status_text}"
    )


def generate_resume_feedback(resume_text, job_description, api_key):
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"""Review the resume against the job description and return only valid JSON with exactly these keys:
"relevant_skills_found": an array of skills explicitly supported by the resume text,
"possibly_missing_skills": an array of job-description skills not found in the resume text,
"improvement_suggestions": an array of specific, truthful resume-editing suggestions.

Rules:
- Treat the resume as the only evidence for claimed skills and experience.
- Do not invent qualifications, skills, projects, or experience.
- A possibly missing skill means only that it was not found in the supplied text; do not imply the person lacks it.
- Suggestions may recommend clarifying or adding evidence only when it is true; never fabricate achievements or metrics.
- Make suggestions specific to the resume and job description. Cover relevant missing keywords (clearly conditional on truth), improving experience bullets, and selecting relevant completed projects when applicable.
- Keep each array concise and relevant. Use empty arrays when appropriate.

RESUME TEXT:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "relevant_skills_found": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "possibly_missing_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "improvement_suggestions": {"type": "ARRAY", "items": {"type": "STRING"}},
                        },
                        "required": [
                            "relevant_skills_found",
                            "possibly_missing_skills",
                            "improvement_suggestions",
                        ],
                    },
                ),
            )
            break
        except Exception as error:
            if getattr(error, "code", None) != 503 or attempt == 1:
                raise
            time.sleep(5)
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    feedback = json.loads(response.text)
    expected_keys = {
        "relevant_skills_found",
        "possibly_missing_skills",
        "improvement_suggestions",
    }
    if set(feedback) != expected_keys or any(
        not isinstance(feedback[key], list)
        or any(not isinstance(item, str) for item in feedback[key])
        for key in expected_keys
    ):
        raise ValueError("Gemini returned feedback in an unexpected format.")
    return feedback


st.set_page_config(
    page_title="AI Resume Success Predictor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #142544;
        --muted: #71809a;
        --blue: #eaf4ff;
        --purple: #7655e8;
        --border: #e5ebf3;
    }

    .stApp {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
        background: linear-gradient(145deg, #e7f2ff 0%, #f4f8ff 35%, #ffffff 78%);
    }

    .stApp .block-container {
        max-width: 2048px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        padding: 1.5rem clamp(1.1rem, 2.4vw, 2.5rem) 1.25rem;
    }

    header[data-testid="stHeader"] { background: transparent; }
    .stApp [data-testid="stHorizontalBlock"] { min-width: 0; }
    .stApp [data-testid="column"] { min-width: 0; }

    .hero { margin: 0 auto 1.25rem; text-align: center; }
    .hero h1 {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.25rem);
        line-height: 1.12;
        font-weight: 800;
        margin: 0 0 .55rem;
    }
    .hero p {
        color: #63738e;
        font-size: 1.08rem;
        margin: 0;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, .94);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 14px 42px rgba(35, 69, 115, .08);
        padding: 1.2rem 1.3rem;
    }

    .section-label {
        display: flex;
        align-items: center;
        gap: .7rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin: .1rem 0 .65rem;
    }
    .step-number {
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 28px;
        border-radius: 50%;
        color: #fff;
        background: #7b61e9;
        font: 700 .82rem 'DM Sans', sans-serif;
    }
    .field-hint {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.5;
        margin: -.3rem 0 .8rem 2.4rem;
    }

    [data-testid="stFileUploader"] section {
        background: #f8fbff;
        border: 1px dashed #bdcce0;
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section > div { padding: .9rem; }
    [data-testid="stFileUploader"] button {
        border: 1px solid #d9e2ef;
        border-radius: 8px;
        color: #40516d;
        background: #fff;
        font-size: .95rem;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small {
        font-size: .95rem;
        line-height: 1.5;
    }
    [data-testid="stTextArea"] textarea {
        min-height: 300px;
        border: 1px solid #dce4ef;
        border-radius: 11px;
        background: #fbfcff;
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.6;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #9a88f0;
        box-shadow: 0 0 0 2px rgba(118, 85, 232, .12);
    }
    .stButton > button[kind="primary"] {
        width: 100%;
        min-height: 3.15rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(100deg, #7655e8, #6848d7);
        color: white;
        font-weight: 700;
        box-shadow: 0 7px 16px rgba(105, 72, 215, .2);
        transition: transform .16s ease, box-shadow .16s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(105, 72, 215, .26);
        background: linear-gradient(100deg, #6d4cdd, #5e3dcc);
        color: white;
    }
    .stButton > button[kind="primary"]:focus-visible,
    [data-testid="stFileUploader"] button:focus-visible,
    [data-testid="stTextArea"] textarea:focus-visible {
        outline: 3px solid #33228f;
        outline-offset: 3px;
    }

    .results-heading {
        color: var(--ink);
        font: 800 1.45rem 'Manrope', sans-serif;
        margin: .05rem 0 1.2rem;
    }
    .empty-state {
        min-height: 235px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 1.6rem;
        border: 1px dashed #cfd9e8;
        border-radius: 14px;
        background: linear-gradient(135deg, #f7faff, #fbfcff);
        margin-bottom: 1.3rem;
    }
    .empty-icon {
        display: grid;
        place-items: center;
        width: 64px;
        height: 64px;
        border-radius: 18px;
        background: #eeeaff;
        color: #795ce2;
        font-size: 1.8rem;
        margin-bottom: .8rem;
    }
    .empty-state h3 {
        color: #34445f;
        font: 700 1.05rem 'Manrope', sans-serif;
        margin: 0 0 .35rem;
    }
    .empty-state p {
        width: 100%;
        max-width: 100%;
        text-align: left;
        color: #7a879b;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
        overflow-wrap: anywhere;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1.1rem;
    }
    .feature {
        --card-accent: #7054d8;
        --card-tint: #f7f4ff;
        min-width: 0;
        padding: 1.25rem;
        border: 1px solid color-mix(in srgb, var(--card-accent) 20%, #e8edf4);
        border-radius: 16px;
        background: linear-gradient(145deg, #fff, var(--card-tint));
        box-shadow: 0 8px 22px rgba(35, 69, 115, .055);
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
    }
    .feature:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--card-accent) 42%, #e8edf4);
        box-shadow: 0 12px 26px rgba(35, 69, 115, .1);
    }
    .feature--skills { --card-accent: #7054d8; --card-tint: #f7f4ff; }
    .feature--suggestions { --card-accent: #118477; --card-tint: #f0faf7; }
    .feature--ats { --card-accent: #3878bd; --card-tint: #f1f7ff; }
    .feature--readiness { --card-accent: #c07c24; --card-tint: #fff8ed; }
    .st-key-result_card_skills { --card-accent: #7054d8; --card-tint: #f7f4ff; }
    .st-key-result_card_suggestions { --card-accent: #118477; --card-tint: #f0faf7; }
    .st-key-result_card_ats { --card-accent: #3878bd; --card-tint: #f1f7ff; }
    .st-key-result_card_readiness { --card-accent: #c07c24; --card-tint: #fff8ed; }
    .st-key-result_card_skills,
    .st-key-result_card_suggestions,
    .st-key-result_card_ats,
    .st-key-result_card_readiness {
        width: 100%;
        min-width: 0;
    }
    .st-key-result_card_skills [data-testid="stButton"] > button,
    .st-key-result_card_suggestions [data-testid="stButton"] > button,
    .st-key-result_card_ats [data-testid="stButton"] > button,
    .st-key-result_card_readiness [data-testid="stButton"] > button {
        width: 100%;
        min-height: 118px;
        min-width: 0;
        box-sizing: border-box;
        border: 1px solid color-mix(in srgb, var(--card-accent) 28%, #e8edf4);
        border-radius: 16px;
        background: linear-gradient(145deg, #fff, var(--card-tint));
        color: #253651;
        padding: 1.25rem;
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
        font-size: 1.08rem;
        line-height: 1.6;
        box-shadow: 0 8px 22px rgba(35, 69, 115, .055);
        transition: border-color .16s ease, box-shadow .16s ease;
    }
    .st-key-result_card_skills [data-testid="stButton"] > button:hover,
    .st-key-result_card_suggestions [data-testid="stButton"] > button:hover,
    .st-key-result_card_ats [data-testid="stButton"] > button:hover,
    .st-key-result_card_readiness [data-testid="stButton"] > button:hover {
        border-color: color-mix(in srgb, var(--card-accent) 52%, #e8edf4);
        box-shadow: 0 12px 26px rgba(35, 69, 115, .1);
        background: linear-gradient(145deg, #fff, var(--card-tint));
        color: #253651;
    }
    .st-key-result_card_skills [data-testid="stButton"] > button:focus-visible,
    .st-key-result_card_suggestions [data-testid="stButton"] > button:focus-visible,
    .st-key-result_card_ats [data-testid="stButton"] > button:focus-visible,
    .st-key-result_card_readiness [data-testid="stButton"] > button:focus-visible {
        outline: 3px solid var(--card-accent);
        outline-offset: 3px;
    }
    .st-key-result_details_skills,
    .st-key-result_details_suggestions,
    .st-key-result_details_ats,
    .st-key-result_details_readiness {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
        padding: 1.25rem;
        border: 1px solid color-mix(in srgb, var(--card-accent) 24%, #e8edf4);
        border-radius: 14px;
        background: linear-gradient(145deg, #fff, var(--card-tint));
        overflow-wrap: anywhere;
    }
    .st-key-result_details_skills { --card-accent: #7054d8; --card-tint: #f7f4ff; }
    .st-key-result_details_suggestions { --card-accent: #118477; --card-tint: #f0faf7; }
    .st-key-result_details_ats { --card-accent: #3878bd; --card-tint: #f1f7ff; }
    .st-key-result_details_readiness { --card-accent: #c07c24; --card-tint: #fff8ed; }
    .feature-header {
        display: flex;
        align-items: center;
        gap: .9rem;
        margin-bottom: 1.15rem;
    }
    .feature-heading { min-width: 0; }
    .feature-kicker, .card-eyebrow {
        display: block;
        color: var(--card-accent);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .09em;
        line-height: 1.5;
    }
    .feature-icon {
        display: grid;
        place-items: center;
        width: 40px;
        height: 40px;
        flex: 0 0 40px;
        border-radius: 12px;
        color: var(--card-accent);
        background: color-mix(in srgb, var(--card-accent) 11%, white);
        font-size: 1.1rem;
    }
    .feature h3 {
        color: #253651;
        font: 700 1.1rem 'Manrope', sans-serif;
        margin: .18rem 0 0;
    }
    .feature-content {
        width: 100%;
        max-width: none;
        min-width: 0;
        color: #495a73;
        font-size: 1rem;
        line-height: 1.7;
        overflow-wrap: anywhere;
    }
    .score-overview {
        display: flex;
        align-items: center;
        min-width: 0;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .score-ring {
        position: relative;
        display: grid;
        place-items: center;
        width: 104px;
        height: 104px;
        flex: 0 0 104px;
        border-radius: 50%;
        background: conic-gradient(var(--card-accent) var(--score), #e8edf5 0);
    }
    .score-ring::before {
        position: absolute;
        inset: 9px;
        border-radius: 50%;
        background: #fff;
        content: "";
    }
    .score-ring-value {
        position: relative;
        color: #253651;
        font: 800 1.35rem 'Manrope', sans-serif;
    }
    .score-copy { display: grid; min-width: 0; gap: .4rem; }
    .score-copy strong { color: #253651; font-size: 1rem; line-height: 1.55; }
    .card-section { margin-top: 1.1rem; }
    .card-section-title {
        display: block;
        color: #34445f;
        font-size: .95rem;
        line-height: 1.5;
        margin-bottom: .55rem;
    }
    .badge-list {
        display: flex;
        flex-wrap: wrap;
        gap: .4rem;
    }
    .skill-badge, .source-badge {
        display: inline-flex;
        max-width: 100%;
        min-width: 0;
        padding: .36rem .68rem;
        border: 1px solid transparent;
        border-radius: 999px;
        font-size: .9rem;
        font-weight: 600;
        line-height: 1.55;
        overflow-wrap: anywhere;
        white-space: normal;
    }
    .skill-badge--matched, .skill-badge--strength, .skill-badge--passed {
        color: #126a5e;
        border-color: #c7e9df;
        background: #e9f8f3;
    }
    .skill-badge--missing, .skill-badge--gap {
        color: #a3443d;
        border-color: #f0d2cc;
        background: #fff1ee;
    }
    .skill-badge--warning {
        color: #965c12;
        border-color: #f1dfb9;
        background: #fff6e5;
    }
    .source-badge {
        color: #11685f;
        border-color: #c7e9df;
        background: #e9f8f3;
    }
    .card-list {
        display: grid;
        gap: .75rem;
        margin: .45rem 0 0;
        padding-left: 1.1rem;
        line-height: 1.65;
    }
    .card-list li {
        min-width: 0;
        padding-left: .12rem;
        overflow-wrap: anywhere;
    }
    .card-empty, .card-note {
        display: block;
        color: #71809a;
        font-size: .9rem;
        line-height: 1.65;
        margin: .75rem 0 0;
    }
    .card-empty { margin-top: .1rem; }

    .footer {
        text-align: center;
        color: #8793a6;
        font-size: .79rem;
        margin-top: 1.7rem;
    }

    @media (max-width: 760px) {
        .block-container { padding: 1.25rem .9rem 1rem; }
        .stApp [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        .stApp [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        .hero { margin-bottom: 1rem; }
        .hero p { font-size: .98rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { padding: 1.15rem; }
        .feature-grid { grid-template-columns: 1fr; }
        .score-overview { align-items: flex-start; }
        .score-ring { width: 88px; height: 88px; flex-basis: 88px; }
        .empty-state { min-height: 200px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>AI Resume Success Predictor</h1>
      <p>See how your resume aligns with your next opportunity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_column, results_column = st.columns([1, 1.2], gap="medium")

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_error" not in st.session_state:
    st.session_state.resume_error = ""
if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "match_error" not in st.session_state:
    st.session_state.match_error = ""
if "gemini_feedback" not in st.session_state:
    st.session_state.gemini_feedback = None
if "keyword_feedback" not in st.session_state:
    st.session_state.keyword_feedback = None
if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = ""
if "feedback_source" not in st.session_state:
    st.session_state.feedback_source = ""
if "ats_report" not in st.session_state:
    st.session_state.ats_report = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "analysis_resume_text" not in st.session_state:
    st.session_state.analysis_resume_text = None
if "analysis_job_description" not in st.session_state:
    st.session_state.analysis_job_description = None
if "active_result_card" not in st.session_state:
    st.session_state.active_result_card = None

with input_column:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label"><span class="step-number">1</span>Resume Upload</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="field-hint">Upload your resume as a PDF</div>', unsafe_allow_html=True)
        resume_file = st.file_uploader(
            "Choose a PDF resume",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if resume_file is not None:
            try:
                with fitz.open(stream=resume_file.getvalue(), filetype="pdf") as resume_pdf:
                    extracted_text = "\n".join(page.get_text() for page in resume_pdf).strip()
                if extracted_text:
                    st.session_state.resume_text = extracted_text
                    st.session_state.resume_error = ""
                else:
                    st.session_state.resume_text = ""
                    st.session_state.resume_error = (
                        "We couldn't find readable text in this PDF. It may be empty or scanned."
                    )
            except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, ValueError):
                st.session_state.resume_text = ""
                st.session_state.resume_error = (
                    "We couldn't read this PDF. Please try a valid, text-based PDF file."
                )
            except Exception:
                st.session_state.resume_text = ""
                st.session_state.resume_error = (
                    "Something went wrong while reading this PDF. Please try another file."
                )
            if st.session_state.resume_error:
                st.warning(st.session_state.resume_error)
        else:
            st.session_state.resume_text = ""
            st.session_state.resume_error = ""

        st.markdown(
            '<div class="section-label" style="margin-top:1.25rem"><span class="step-number">2</span>Job Description</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="field-hint">Paste the job description you are targeting</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Job description",
            placeholder="Paste the job description here...",
            height=300,
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("Analyze Resume", type="primary", use_container_width=True)
        if analyze_clicked:
            st.session_state.active_result_card = None
            st.session_state.analysis_resume_text = st.session_state.resume_text
            st.session_state.analysis_job_description = job_description
            st.session_state.match_result = None
            st.session_state.match_error = ""
            st.session_state.gemini_feedback = None
            st.session_state.keyword_feedback = None
            st.session_state.gemini_error = ""
            st.session_state.feedback_source = ""
            st.session_state.ats_report = None
            st.session_state.analysis_complete = False
            if not st.session_state.resume_text.strip():
                st.session_state.match_error = "Please upload a readable PDF resume first."
            elif not job_description.strip():
                st.session_state.match_error = "Please enter a job description to compare with your resume."
            else:
                st.session_state.analysis_complete = True
                local_feedback = generate_local_resume_feedback(
                    st.session_state.resume_text,
                    job_description,
                )
                st.session_state.keyword_feedback = local_feedback
                st.session_state.gemini_feedback = local_feedback
                st.session_state.feedback_source = "local"
                st.session_state.ats_report = assess_ats_compatibility(
                    st.session_state.resume_text
                )
                try:
                    st.session_state.match_result = calculate_resume_match(
                        st.session_state.resume_text,
                        job_description,
                    )
                except ValueError:
                    st.session_state.match_result = None
                    st.session_state.match_error = (
                        "Text similarity could not be calculated from these terms; "
                        "keyword feedback and basic checks are still available."
                    )
                except Exception:
                    st.session_state.match_result = None
                    st.session_state.match_error = (
                        "Text similarity could not be calculated; keyword feedback "
                        "and basic checks are still available."
                    )
                api_key = os.getenv("GEMINI_API_KEY", "").strip()
                if not api_key:
                    st.session_state.gemini_error = (
                        "Gemini feedback is unavailable. Add GEMINI_API_KEY to your .env file and restart the app."
                    )
                elif st.session_state.match_result is not None:
                    try:
                        st.session_state.gemini_feedback = generate_resume_feedback(
                            st.session_state.resume_text,
                            job_description,
                            api_key,
                        )
                        st.session_state.feedback_source = "gemini"
                    except Exception as error:
                        print(_safe_gemini_error_diagnostic(error))
                        st.session_state.gemini_error = (
                            "Gemini feedback couldn't be generated right now. Please check your API configuration and try again."
                        )

with results_column:
    with st.container(border=True):
        st.markdown('<div class="results-heading">Your Results</div>', unsafe_allow_html=True)
        results_are_current = (
            st.session_state.analysis_resume_text == st.session_state.resume_text
            and st.session_state.analysis_job_description == job_description
        )
        if st.session_state.analysis_complete and results_are_current:
            if st.session_state.match_result is not None:
                similarity_percentage, _ = st.session_state.match_result
                similarity_text = (
                    f"<strong>Text similarity (TF-IDF cosine):</strong> "
                    f"{similarity_percentage}% — weighted word/phrase overlap, "
                    "not a job-success probability."
                )
            else:
                similarity_text = html.escape(
                    st.session_state.match_error
                    or "Text similarity could not be calculated."
                )
            feature_card_content = build_feature_card_content(
                st.session_state.keyword_feedback,
                st.session_state.gemini_feedback,
                st.session_state.ats_report,
                st.session_state.feedback_source,
            )
            result_title = "Resume Analysis Complete"
            result_description = [similarity_text]
            if st.session_state.gemini_feedback:
                feedback_source = (
                    "Gemini-generated suggestions; keyword and ATS checks are local"
                    if st.session_state.feedback_source == "gemini"
                    else "Local keyword-based feedback (Gemini unavailable)"
                )
                result_description.append(
                    f"<strong>Feedback source:</strong> {feedback_source}"
                )
            elif st.session_state.gemini_error:
                result_description.append(html.escape(st.session_state.gemini_error))
            result_description.append(
                "Select a result card below to inspect its detailed findings."
            )
            result_description = "<br><br>".join(result_description)
        elif results_are_current and st.session_state.match_error:
            result_title = "Unable to compare yet"
            result_description = html.escape(st.session_state.match_error)
        else:
            result_title = "Your insights will appear here"
            result_description = "Upload your resume and add a job description to see your personalized results."

        st.markdown(
            f"""
            <div class="empty-state">
              <div class="empty-icon">✦</div>
              <h3>{result_title}</h3>
              <p>{result_description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        card_rows = (
            (
                (
                    "skill_match",
                    "Skill Match Score",
                    "◉",
                    "View matched and missing skills with the score explanation.",
                    "skills",
                ),
                (
                    "improvement_suggestions",
                    "Improvement Suggestions",
                    "✎",
                    "View the personalized resume recommendations.",
                    "suggestions",
                ),
            ),
            (
                (
                    "ats_compatibility",
                    "ATS Compatibility",
                    "✓",
                    "View passed checks, warnings, and fixes.",
                    "ats",
                ),
                (
                    "job_readiness",
                    "Be Job Ready",
                    "↗",
                    "View strengths, gaps, and suggested next actions.",
                    "readiness",
                ),
            ),
        )
        for row in card_rows:
            card_columns = st.columns(2, gap="medium")
            for column, (card_id, title, icon, description, style_name) in zip(
                card_columns, row
            ):
                with column:
                    with st.container(key=f"result_card_{style_name}"):
                        selected = (
                            st.session_state.active_result_card == card_id
                        )
                        st.button(
                            f"{icon}  **{title}**  \n{description}  \n"
                            f"{'⌄ Expanded' if selected else '› Expand'}",
                            use_container_width=True,
                            key=f"result_toggle_{card_id}",
                            on_click=_toggle_result_card,
                            args=(card_id,),
                        )

        selected_card = next(
            (
                card
                for row in card_rows
                for card in row
                if card[0] == st.session_state.active_result_card
            ),
            None,
        )
        if selected_card is not None:
            card_id, _, _, _, style_name = selected_card
            with st.container(key=f"result_details_{style_name}"):
                if st.session_state.analysis_complete and results_are_current:
                    st.markdown(
                        f'<div class="feature-content">'
                        f'{feature_card_content[card_id]}'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(
                        "Analyze the current resume and job "
                        "description to view up-to-date details."
                    )

st.markdown('<div class="footer">Powered by Generative AI &bull; Built with Streamlit</div>', unsafe_allow_html=True)
