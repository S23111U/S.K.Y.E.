"""One-time Google Calendar login. Opens a browser for consent and stores the
token in memory/google_token.json. Run again if the login ever expires."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server"))
import calendar_tools as ct
from google_auth_oauthlib.flow import InstalledAppFlow

if not os.path.exists(ct.KEYS_FILE):
    sys.exit(f"Missing {ct.KEYS_FILE}")
creds = InstalledAppFlow.from_client_secrets_file(ct.KEYS_FILE, ct.SCOPES).run_local_server(port=0, prompt="consent", open_browser=False, timeout_seconds=600)
os.makedirs(os.path.dirname(ct.TOKEN_FILE), exist_ok=True)
with open(ct.TOKEN_FILE, "w") as f:
    f.write(creds.to_json())
print("Logged in. Token saved to", ct.TOKEN_FILE)
