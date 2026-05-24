import base64
import os
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _decode_base64url(data: str) -> str:
    if not data:
        return ""

    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="ignore")


def _extract_text_from_payload(payload: Dict) -> str:
    if not payload:
        return ""

    body = payload.get("body", {})
    data = body.get("data")
    if data:
        return _decode_base64url(data)

    parts = payload.get("parts", [])
    for part in parts:
        mime_type = part.get("mimeType", "")
        if mime_type == "text/plain":
            part_data = part.get("body", {}).get("data")
            if part_data:
                return _decode_base64url(part_data)

        nested = _extract_text_from_payload(part)
        if nested:
            return nested

    return ""


def _extract_headers(payload: Dict) -> Dict[str, str]:
    headers = {}
    for header in payload.get("headers", []):
        name = header.get("name", "")
        value = header.get("value", "")
        if name:
            headers[name.lower()] = value
    return headers


def _build_credentials(credentials_path: str, token_path: str) -> Credentials:
    creds: Optional[Credentials] = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Missing Gmail credentials file: {credentials_path}. Place a Google OAuth client JSON file there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return creds


def load_gmail_messages(
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
    query: str = "in:inbox newer_than:7d",
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """Load Gmail messages using Gmail API with read-only OAuth access."""
    creds = _build_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)

    response = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results,
    ).execute()

    items = response.get("messages", [])
    messages: List[Dict[str, str]] = []

    for item in items:
        detail = service.users().messages().get(
            userId="me",
            id=item["id"],
            format="full",
        ).execute()

        payload = detail.get("payload", {})
        headers = _extract_headers(payload)
        body = _extract_text_from_payload(payload) or detail.get("snippet", "")

        messages.append({
            "from": headers.get("from", "unknown@gmail.com"),
            "subject": headers.get("subject", "No Subject"),
            "body": body,
        })

    return messages