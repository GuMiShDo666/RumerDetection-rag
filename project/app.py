import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Suppress OTel "Failed to detach context" warning caused by generator/context interaction.
# Tracing is unaffected.
# Known bug: https://github.com/open-telemetry/opentelemetry-python/issues/2606
class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

def _launch_auth():
    username = os.environ.get("RAG_AUTH_USERNAME", "").strip()
    password = os.environ.get("RAG_AUTH_PASSWORD", "").strip()
    if username and password:
        return (username, password)
    if password:
        return ("admin", password)
    return None

def _server_host():
    return os.environ.get("RAG_SERVER_NAME", "127.0.0.1")

def _server_port():
    return int(os.environ.get("RAG_SERVER_PORT", "7860"))

def _launch_html():
    import uvicorn
    from web_app import create_web_app

    print("\nCreating RumorDetection RAG web app...")
    app = create_web_app()
    print("\nLaunching RumorDetection RAG...")
    uvicorn.run(app, host=_server_host(), port=_server_port())

if __name__ == "__main__":
    _launch_html()
