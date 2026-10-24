import re


GENERIC_JOB_TERMS = {
    "ability", "candidate", "candidates", "developer", "developers", "engineer",
    "engineers", "experience", "experienced", "job", "knowledge", "manager", "managers",
    "position", "positions", "preferred", "professional", "qualification", "qualifications",
    "requirement", "requirements", "responsibilities", "responsibility", "role", "roles",
    "skill", "skills", "strong", "team", "teams", "work", "working", "year", "years",
    "basic", "entry", "fresher", "junior", "level", "senior", "required",
    "it", "mi", "title", "welcome", "recent", "problem",
}

SKILL_CATALOG = (
    ("java", "Java", ("java", "core java")),
    ("javascript", "JavaScript", ("javascript",)),
    ("python", "Python", ("python",)),
    ("sql", "SQL", ("sql",)),
    ("mysql", "MySQL", ("mysql",)),
    ("postgresql", "PostgreSQL", ("postgresql", "postgres")),
    ("mongodb", "MongoDB", ("mongodb", "mongo db")),
    ("redis", "Redis", ("redis",)),
    ("c++", "C++", ("c++",)),
    ("c#", "C#", ("c#",)),
    ("typescript", "TypeScript", ("typescript",)),
    ("go", "Go", ("golang", "go language", "go programming language")),
    ("rust", "Rust", ("rust",)),
    ("php", "PHP", ("php",)),
    ("ruby", "Ruby", ("ruby",)),
    (".net", ".NET", (".net", "dotnet")),
    ("html", "HTML", ("html",)),
    ("css", "CSS", ("css",)),
    ("react", "React", ("react", "react.js")),
    ("angular", "Angular", ("angular",)),
    ("vue", "Vue", ("vue", "vue.js")),
    ("node.js", "Node.js", ("node.js", "node js")),
    ("express", "Express", ("express", "express.js")),
    ("django", "Django", ("django",)),
    ("flask", "Flask", ("flask",)),
    ("fastapi", "FastAPI", ("fastapi", "fast api")),
    ("spring", "Spring", ("spring", "spring boot")),
    ("rest_api", "REST APIs", ("rest api", "rest apis", "restful api", "restful apis")),
    ("graphql", "GraphQL", ("graphql",)),
    ("api", "APIs", ("api", "apis")),
    ("git", "Git", ("git",)),
    ("docker", "Docker", ("docker",)),
    ("kubernetes", "Kubernetes", ("kubernetes", "k8s")),
    ("aws", "AWS", ("aws", "amazon web services")),
    ("azure", "Azure", ("azure",)),
    ("gcp", "Google Cloud", ("gcp", "google cloud")),
    ("linux", "Linux", ("linux",)),
    ("machine_learning", "Machine Learning", ("machine learning",)),
    ("deep_learning", "Deep Learning", ("deep learning",)),
    ("data_science", "Data Science", ("data science",)),
    ("data_analysis", "Data Analysis", ("data analysis", "data analytics")),
    ("natural_language_processing", "Natural Language Processing", ("natural language processing", "nlp")),
    ("computer_vision", "Computer Vision", ("computer vision",)),
    ("tensorflow", "TensorFlow", ("tensorflow",)),
    ("pytorch", "PyTorch", ("pytorch",)),
    ("pandas", "Pandas", ("pandas",)),
    ("numpy", "NumPy", ("numpy",)),
    ("spark", "Apache Spark", ("apache spark", "spark")),
    ("communication", "Communication", ("communication", "written communication", "verbal communication")),
    ("problem_solving", "Problem-Solving", ("problem solving",)),
    ("collaboration", "Collaboration", ("collaboration", "teamwork")),
    ("leadership", "Leadership", ("leadership",)),
    ("project_management", "Project Management", ("project management",)),
    ("agile", "Agile", ("agile",)),
    ("scrum", "Scrum", ("scrum",)),
    ("testing", "Software Testing", ("software testing", "testing")),
    ("unit_testing", "Unit Testing", ("unit testing",)),
    ("automation", "Test Automation", ("test automation", "automated testing")),
    ("cybersecurity", "Cybersecurity", ("cybersecurity", "cyber security")),
    ("networking", "Networking", ("computer networking", "networking")),
    ("oop", "Object-Oriented Programming (OOP)", ("object oriented programming", "oop")),
)


def _normalize_keyword(text):
    return " ".join(re.findall(r"[a-z0-9+#]+", text.lower()))


_SKILL_BY_KEY = {
    key: (label, aliases) for key, label, aliases in SKILL_CATALOG
}
_SKILL_KEY_BY_ALIAS = {
    _normalize_keyword(alias): key
    for key, _, aliases in SKILL_CATALOG
    for alias in aliases
}


def _contains_phrase(text, phrase):
    normalized_text = f" {_normalize_keyword(text)} "
    normalized_phrase = f" {_normalize_keyword(phrase)} "
    return normalized_phrase in normalized_text


def _skill_key(term):
    normalized = _normalize_keyword(term)
    if normalized in _SKILL_KEY_BY_ALIAS:
        return _SKILL_KEY_BY_ALIAS[normalized]
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
    if normalized in {"javascript", "js"}:
        return "javascript"
    return normalized


def _keyword_is_present(keyword, text):
    key = _skill_key(keyword)
    if key in _SKILL_BY_KEY:
        _, aliases = _SKILL_BY_KEY[key]
        return any(_contains_phrase(text, alias) for alias in aliases) or (
            key == "javascript" and _contains_phrase(text, "js")
        )
    if key == "java":
        return _contains_phrase(text, "java")
    if key == "sql":
        return _contains_phrase(text, "sql")
    if key == "mysql":
        return _contains_phrase(text, "mysql")
    if key == "oop":
        return _contains_phrase(text, "oop") or _contains_phrase(
            text, "object oriented programming"
        )
    if key == "javascript":
        return _contains_phrase(text, "javascript") or _contains_phrase(text, "js")
    return _contains_phrase(text, keyword)


def _recognized_job_skills(job_description):
    recognized = [
        (key, label)
        for key, label, aliases in SKILL_CATALOG
        if any(_contains_phrase(job_description, alias) for alias in aliases)
    ]
    if any(key == "rest_api" for key, _ in recognized):
        recognized = [(key, label) for key, label in recognized if key != "api"]
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


def calculate_keyword_match(resume_text, job_description):
    recognized_skills = _recognized_job_skills(job_description)
    detected_skills = [label for _, label in recognized_skills]

    matched_skills = [
        skill for skill in detected_skills
        if _keyword_is_present(skill, resume_text)
    ]
    missing_skills = [
        skill for skill in detected_skills
        if not _keyword_is_present(skill, resume_text)
    ]
    percentage = (
        round(len(matched_skills) / len(detected_skills) * 100, 1)
        if detected_skills
        else None
    )
    return {
        "detected_skills": detected_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "percentage": percentage,
    }
