"""
Email notification via Resend API for the Brokerage Model.

This version uses direct HTTP calls (no extra 'resend' SDK dependency)
and preserves your original robust error handling, HTML monospace formatting,
and failure alert functionality.
"""

import os
import json
import html as html_lib
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


def _wrap_as_html(plain_body: str) -> str:
    """
    Wrap a plain-text body in HTML with monospace styling so column
    alignment is preserved across email clients.
    """
    escaped = html_lib.escape(plain_body)
    
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ margin: 0; padding: 16px; background: #f8f8f8; }}
  pre {{
    font-family: 'SF Mono', Menlo, Monaco, Consolas, 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.4;
    white-space: pre;
    background: #ffffff;
    color: #1a1a1a;
    padding: 16px;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    margin: 0;
    overflow-x: auto;
  }}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>"""


def send_email(
    subject: str = None,
    body: str = None,
    to_email: str = None,
    from_email: str = None,
) -> bool:
    """
    Send the monthly signal report via Resend.
    
    If subject and body are not provided, it will use the default Brokerage Model subject.
    """
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logger.error("RESEND_API_KEY environment variable not set")
        print("RESEND_API_KEY not set — printing report to console instead:")
        print("\n" + "=" * 60)
        print(body or "(no body provided)")
        print("=" * 60)
        return False

    to = to_email or os.environ.get("TAA_EMAIL_TO")
    sender = from_email or os.environ.get("TAA_EMAIL_FROM", "Brokerage Model <signals@resend.dev>")

    if not to:
        logger.error("No recipient email configured (TAA_EMAIL_TO)")
        return False

    # Default subject for the Brokerage Model
    if subject is None:
        subject = f"Brokerage Model — Monthly Signal Report {datetime.now().strftime('%Y-%m-%d')}"

    payload = json.dumps({
        "from": sender,
        "to": [to],
        "subject": subject,
        "text": body,                    # plain-text fallback
        "html": _wrap_as_html(body),     # formatted HTML version
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Brokerage-Model/1.0 (Python; +https://github.com)",
        "Accept": "application/json",
    }

    logger.info(f"Sending email: from={sender!r}, to={to!r}, subject={subject!r}")

    try:
        req = Request(RESEND_API_URL, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            logger.info(f"Email sent successfully. ID: {result.get('id', 'unknown')}")
            return True
    except HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = "(could not read error body)"
        logger.error(
            f"Failed to send email: HTTP {e.code} {e.reason}\n"
            f"Resend response: {error_body}"
        )
        return False
    except URLError as e:
        logger.error(f"Failed to send email (network error): {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
        return False


def send_failure_alert(error_message: str, traceback_str: str = ""):
    """
    Send a failure notification so you know the script broke.
    """
    subject = "[Brokerage Model] ⚠️ SIGNAL COMPUTATION FAILED"
    body = (
        "The monthly Brokerage Model signal computation failed.\n\n"
        f"Error: {error_message}\n\n"
        "DO NOT TRADE until this is resolved.\n"
        "Check the GitHub Actions log for full details.\n\n"
    )
    if traceback_str:
        body += f"Traceback:\n{traceback_str}\n"
    
    send_email(subject, body)
