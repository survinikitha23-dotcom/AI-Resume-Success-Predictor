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


load_dotenv()


GENERIC_JOB_TERMS = {
    "ability", "candidate", "candidates", "developer", "developers", "engineer",
    "engineers", "experience", "experienced", "job", "knowledge", "manager", "managers",
    "position", "positions", "preferred", "professional", "qualification", "qualifications",
    "requirement", "requirements", "responsibilities", "responsibility", "role", "roles",
    "skill", "skills", "strong", "team", "teams", "work", "working", "year", "years",
}


def _normalize_keyword(text):
    return " ".join(re.findall(r"[a-z0-9+#]+", text.lower()))


def _contains_phrase(text, phrase):
    normalized_text = f" {_normalize_keyword(text)} "
    normalized_phrase = f" {_normalize_keyword(phrase)} "
    return normalized_phrase in normalized_text


def _skill_key(term):
    normalized = _normalize_keyword(term)
    if normalized in {"java", "core java"}:
        return "java"
    if normalized in {"sql", "mysql"}:
        return normalized
    if normalized in {
        "oop",
        "object oriented programming",
        "object oriented programming oop",
    }:
        return "oop"
    if normalized == "javascript":
        return "javascript"
    return normalized


def _keyword_is_present(keyword, text):
    key = _skill_key(keyword)
    if key == "java":
        return _contains_phrase(text, "java")
    if key == "sql":
        return _contains_phrase(text, "sql") or _contains_phrase(text, "mysql")
    if key == "mysql":
        return _contains_phrase(text, "mysql")
    if key == "oop":
        return _contains_phrase(text, "oop") or _contains_phrase(
            text, "object oriented programming"
        )
    if key == "javascript":
        return _contains_phrase(text, "javascript")
    return _contains_phrase(text, keyword)


def _recognized_job_skills(job_description):
    recognized = []

    def add_skill(key, label, phrase):
        if _contains_phrase(job_description, phrase) and all(
            existing_key != key for existing_key, _ in recognized
        ):
            recognized.append((key, label))

    add_skill("java", "Core Java", "core java")
    add_skill("java", "Java", "java")
    add_skill("mysql", "MySQL", "mysql")
    add_skill("sql", "SQL", "sql")
    add_skill(
        "oop",
        "Object-Oriented Programming (OOP)",
        "object oriented programming",
    )
    add_skill("oop", "OOP", "oop")
    add_skill("javascript", "JavaScript", "javascript")
    return recognized


def _is_recognized_skill_fragment(term, recognized_skills):
    normalized_term = _normalize_keyword(term)
    term_words = set(normalized_term.split())
    for key, label in recognized_skills:
        if _skill_key(term) == key:
            return True
        skill_words = set(_normalize_keyword(label).split())
        if term_words and term_words.issubset(skill_words):
            return True
    return False


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
    documents = [resume_text.strip(), job_description.strip()]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    try:
        document_vectors = vectorizer.fit_transform(documents)
    except ValueError:
        return {
            "relevant_skills_found": [],
            "possibly_missing_skills": [],
            "improvement_suggestions": [
                "Add specific skills or responsibilities to the job description "
                "to enable a keyword comparison.",
                "Tailor your summary and experience bullets to the role using only "
                "information that accurately reflects your background.",
            ],
        }
    terms = vectorizer.get_feature_names_out()
    job_weights = document_vectors[1].toarray().ravel()
    resume_lower = documents[0].lower()
    ranked_terms = sorted(
        range(len(terms)),
        key=lambda index: (" " not in terms[index], -job_weights[index]),
    )

    matching_terms = []
    missing_terms = []
    recognized_skills = _recognized_job_skills(job_description)
    for key, label in recognized_skills:
        destination = (
            matching_terms
            if _keyword_is_present(label, resume_lower)
            else missing_terms
        )
        destination.append(label)

    for index in ranked_terms:
        term = terms[index]
        if job_weights[index] == 0:
            continue
        term_words = term.split()
        if any(word in GENERIC_JOB_TERMS or word == "required" for word in term_words):
            continue
        if _is_recognized_skill_fragment(term, recognized_skills):
            continue
        if any(set(term_words).issubset(existing.split()) for existing in matching_terms):
            continue
        if any(set(term_words).issubset(existing.split()) for existing in missing_terms):
            continue
        destination = (
            matching_terms
            if _keyword_is_present(term, resume_lower)
            else missing_terms
        )
        if len(destination) < 12:
            destination.append(term)

    suggestions = []
    if missing_terms:
        suggestions.append(
            "If accurate, include evidence for these job-description keywords: "
            + ", ".join(missing_terms)
            + "."
        )
    if matching_terms:
        suggestions.append(
            "Strengthen matching keywords with a truthful project, responsibility, "
            "or result from your experience."
        )
    suggestions.append(
        "Tailor your summary and experience bullets to the role using only "
        "information that accurately reflects your background."
    )

    return {
        "relevant_skills_found": matching_terms,
        "possibly_missing_skills": missing_terms,
        "improvement_suggestions": suggestions,
    }


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
- Keep each array concise and relevant. Use empty arrays when appropriate.

RESUME TEXT:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
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

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: linear-gradient(145deg, #e7f2ff 0%, #f4f8ff 35%, #ffffff 78%);
    }

    .block-container {
        max-width: 100%;
        padding: 3rem 2rem 1.5rem;
    }

    header[data-testid="stHeader"] { background: transparent; }

    .hero { margin: 0 auto 2.1rem; text-align: center; }
    .hero h1 {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.25rem);
        line-height: 1.12;
        font-weight: 800;
        margin: 0 0 .8rem;
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
        padding: 1.45rem 1.55rem;
    }

    .section-label {
        display: flex;
        align-items: center;
        gap: .7rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        margin: .1rem 0 .8rem;
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
        font-size: .88rem;
        margin: -.35rem 0 .7rem 2.4rem;
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
    }
    [data-testid="stTextArea"] textarea {
        min-height: 220px;
        border: 1px solid #dce4ef;
        border-radius: 11px;
        background: #fbfcff;
        color: var(--ink);
        font-size: .94rem;
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
        font-size: .9rem;
        line-height: 1.55;
        margin: 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .8rem;
    }
    .feature {
        min-height: 106px;
        display: flex;
        align-items: flex-start;
        gap: .8rem;
        padding: 1rem;
        border: 1px solid #e8edf4;
        border-radius: 12px;
        background: #fff;
    }
    .feature-icon {
        display: grid;
        place-items: center;
        width: 36px;
        height: 36px;
        flex: 0 0 36px;
        border-radius: 10px;
        color: #6651cb;
        background: #f0edff;
        font-size: 1.05rem;
    }
    .feature:nth-child(2) .feature-icon { color: #168477; background: #e7f7f3; }
    .feature:nth-child(3) .feature-icon { color: #3878bd; background: #eaf4ff; }
    .feature:nth-child(4) .feature-icon { color: #c07c24; background: #fff4e4; }
    .feature h3 {
        color: #253651;
        font: 700 .91rem 'Manrope', sans-serif;
        margin: .1rem 0 .3rem;
    }
    .feature p { color: #7a879b; font-size: .81rem; line-height: 1.45; margin: 0; }
    .footer {
        text-align: center;
        color: #8793a6;
        font-size: .79rem;
        margin-top: 1.7rem;
    }

    @media (max-width: 760px) {
        .block-container { padding: 2rem 1rem 1rem; }
        .hero { margin-bottom: 1.5rem; }
        .hero p { font-size: .98rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { padding: 1.15rem; }
        .feature-grid { grid-template-columns: 1fr; }
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

input_column, results_column = st.columns([0.9, 1.25], gap="large")

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
if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = ""
if "feedback_source" not in st.session_state:
    st.session_state.feedback_source = ""

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
            height=220,
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("Analyze Resume", type="primary", use_container_width=True)
        if analyze_clicked:
            st.session_state.match_result = None
            st.session_state.match_error = ""
            st.session_state.gemini_feedback = None
            st.session_state.gemini_error = ""
            st.session_state.feedback_source = ""
            if not st.session_state.resume_text.strip():
                st.session_state.match_error = "Please upload a readable PDF resume first."
            elif not job_description.strip():
                st.session_state.match_error = "Please enter a job description to compare with your resume."
            else:
                try:
                    st.session_state.match_result = calculate_resume_match(
                        st.session_state.resume_text,
                        job_description,
                    )
                except ValueError:
                    st.session_state.match_error = (
                        "We couldn't compare this text. Please add more readable resume and job description text."
                    )
                except Exception:
                    st.session_state.match_error = (
                        "Something went wrong while comparing the text. Please try again."
                    )
                api_key = os.getenv("GEMINI_API_KEY", "").strip()
                if not api_key:
                    st.session_state.gemini_error = (
                        "Gemini feedback is unavailable. Add GEMINI_API_KEY to your .env file and restart the app."
                    )
                    st.session_state.gemini_feedback = generate_local_resume_feedback(
                        st.session_state.resume_text,
                        job_description,
                    )
                    st.session_state.feedback_source = "local"
                else:
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
                        st.session_state.gemini_feedback = generate_local_resume_feedback(
                            st.session_state.resume_text,
                            job_description,
                        )
                        st.session_state.feedback_source = "local"

with results_column:
    with st.container(border=True):
        st.markdown('<div class="results-heading">Your Results</div>', unsafe_allow_html=True)
        if st.session_state.match_result is not None:
            similarity_percentage, matching_terms = st.session_state.match_result
            matching_terms_text = (
                ", ".join(html.escape(term) for term in matching_terms)
                if matching_terms
                else "No matching job-description terms were found in the resume text."
            )
            result_title = "Resume Feedback"
            result_description = [
                f"<strong>Text similarity:</strong> {similarity_percentage}%",
                f"<strong>Matching terms:</strong> {matching_terms_text}",
            ]
            if st.session_state.gemini_feedback:
                feedback_source = (
                    "AI-generated by Gemini"
                    if st.session_state.feedback_source == "gemini"
                    else "Local keyword-based feedback (Gemini unavailable)"
                )
                result_description.append(
                    f"<strong>Feedback source:</strong> {feedback_source}"
                )
                feedback_sections = (
                    ("Relevant skills found in resume", "relevant_skills_found"),
                    ("Possibly missing from resume text", "possibly_missing_skills"),
                    ("Resume improvement suggestions", "improvement_suggestions"),
                )
                for section_title, section_key in feedback_sections:
                    feedback_items = st.session_state.gemini_feedback[section_key]
                    separator = (
                        "<br><br>"
                        if section_key == "improvement_suggestions"
                        else ", "
                    )
                    safe_items = separator.join(
                        html.escape(item) for item in feedback_items
                    )
                    if not safe_items:
                        safe_items = "None identified."
                    result_description.append(
                        f"<strong>{section_title}:</strong> {safe_items}"
                    )
                result_description.append(
                    "<em>Suggested missing skills are not verified as absent; they were not found in the provided resume text.</em>"
                )
            elif st.session_state.gemini_error:
                result_description.append(html.escape(st.session_state.gemini_error))
            result_description = "<br><br>".join(result_description)
        elif st.session_state.match_error:
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
            <div class="feature-grid">
              <div class="feature"><div class="feature-icon">◉</div><div><h3>Skill Match Score</h3><p>See how your skills align with the role.</p></div></div>
              <div class="feature"><div class="feature-icon">✎</div><div><h3>Improvement Suggestions</h3><p>Find opportunities to strengthen your resume.</p></div></div>
              <div class="feature"><div class="feature-icon">✓</div><div><h3>ATS Compatibility</h3><p>Understand how your resume reads for ATS.</p></div></div>
              <div class="feature"><div class="feature-icon">↗</div><div><h3>Be Job Ready</h3><p>Get ready to take the next step in your search.</p></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="footer">Powered by Generative AI &bull; Built with Streamlit</div>', unsafe_allow_html=True)
