import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai


def sanitize_explanation(message, api_key):
    if not isinstance(message, str) or not message.strip():
        return "No server explanation was provided."

    sanitized = message
    if api_key:
        sanitized = re.sub(
            re.escape(api_key),
            "[REDACTED_API_KEY]",
            sanitized,
            flags=re.IGNORECASE,
        )

    sanitized = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(?:AIza[0-9A-Za-z_-]{20,}|ya29\.[A-Za-z0-9._~+/=-]+)\b",
        "[REDACTED_TOKEN]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(?:https?|wss?)://[^\s<>\"']+",
        "[REDACTED_URL]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d+)?(?:/[^\s<>\"']*)?",
        "[REDACTED_URL]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(?:authorization|x-goog-api-key|api[_-]?key|access[_-]?token|"
        r"token|auth|key)"
        r"\s*[:=]\s*[^\s,;]+",
        "[REDACTED_CREDENTIAL]",
        sanitized,
    )
    sanitized = re.sub(r"[?&][^\s<>\"']+", "?[REDACTED_QUERY]", sanitized)
    return " ".join(sanitized.split())


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    print("Error: GEMINI_API_KEY is missing.")
    sys.exit(1)

client = None
try:
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Reply with exactly: Gemini connection successful",
        )
    except Exception as error:
        status_code = getattr(error, "code", None)
        if status_code != 503:
            raise

        print("Gemini is temporarily unavailable; retrying once in 5 seconds.")
        time.sleep(5)
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Reply with exactly: Gemini connection successful",
        )

    if not response.text:
        print("Error: Gemini returned an empty response.")
        sys.exit(1)
    print("Gemini API generation successful.")
except Exception as error:
    response_obj = getattr(error, "response", None)
    http_status = getattr(error, "code", None)
    if not isinstance(http_status, int) or isinstance(http_status, bool):
        http_status = getattr(response_obj, "status_code", None)

    error_status = getattr(error, "status", None)
    if not isinstance(error_status, str) or not error_status.strip():
        error_status = "unavailable"
    else:
        error_status = sanitize_explanation(error_status, api_key)

    explanation = sanitize_explanation(
        getattr(error, "message", None),
        api_key,
    )
    status_output = (
        str(http_status)
        if isinstance(http_status, int) and not isinstance(http_status, bool)
        else "unavailable"
    )
    print(f"HTTP status: {status_output}")
    print(f"Error code/status: {error_status}")
    print(f"Sanitized explanation: {explanation}")
    sys.exit(1)
finally:
    if client is not None:
        client.close()
