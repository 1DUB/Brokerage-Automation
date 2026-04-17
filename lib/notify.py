"""
Email notification via Resend API.

Resend is a transactional email service with a generous free tier.
Sign up at https://resend.com and add your API key as a GitHub secret.
"""

import os
import json
import html as html_lib
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


def _wrap_as_html(plain_body: str) -> str:
    """
    Wrap a plain-text body in HTML with monospace styling so column
    alignment is preserved across email clients.
    
    Most email clients render plain-text emails in proportional fonts,
    which breaks space-based column alignment. Wrapping in <pre> with
    an explicit monospace font-family forces fixed-width rendering.
    """
    # HTML-escape the body so any &, <, > render correctly
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
    subject: str,
    body: str,
    to_email: str = None,
    from_email: str = None,
) -> bool:
    """
    Send an email via Resend, with both HTML (monospace) and plain-text 
    versions. The HTML version preserves column alignment in the report.
    
    Environment variables:
        RESEND_API_KEY: Your Resend API key (required)
        TAA_EMAIL_TO: Recipient email (can be overridden by to_email arg)
        TAA_EMAIL_FROM: Sender email (must be verified in Resend)
    
    Returns:
        True if sent successfully, False otherwise.
    """
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logger.error("RESEND_API_KEY environment variable not set")
        return False
    
    to = to_email or os.environ.get("TAA_EMAIL_TO")
    sender = from_email or os.environ.get("TAA_EMAIL_FROM", "TAA Signals <signals@resend.dev>")
    
    if not to:
        logger.error("No recipient email configured (TAA_EMAIL_TO)")
        return False
    
    payload = json.dumps({
        "from": sender,
        "to": [to],
        "subject": subject,
        "text": body,                # plain-text fallback
        "html": _wrap_as_html(body), # primary rendering with monospace font
    }).encode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Cloudflare (Resend's CDN) blocks default Python-urllib User-Agent
        # as "bad bot" with error 1010. A realistic User-Agent bypasses this.
        "User-Agent": "TAA-Signals/1.0 (Python; +https://github.com)",
        "Accept": "application/json",
    }
    
    # Log what we're sending (without the API key) for debugging
    logger.info(f"Sending email: from={sender!r}, to={to!r}, subject={subject!r}")
    
    try:
        req = Request(RESEND_API_URL, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            logger.info(f"Email sent successfully. ID: {result.get('id', 'unknown')}")
            return True
    except HTTPError as e:
        # Read Resend's error response body — this contains the real error message
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
    subject = "[TAA] ⚠️ SIGNAL COMPUTATION FAILED"
    body = (
        "The monthly TAA signal computation failed.\n\n"
        f"Error: {error_message}\n\n"
        "DO NOT TRADE until this is resolved.\n"
        "Check the GitHub Actions log for details.\n\n"
    )
    if traceback_str:
        body += f"Traceback:\n{traceback_str}\n"
    
    send_email(subject, body)
