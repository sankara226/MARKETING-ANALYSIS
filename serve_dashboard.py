"""Serve outputs/dashboard.html on localhost for a quick browser preview."""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    """Start a local HTTP server rooted at outputs/ and open the dashboard in a browser."""
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(OUTPUTS_DIR), **kwargs)
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print(f"Serving dashboard at {url} (Ctrl+C to stop)")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
