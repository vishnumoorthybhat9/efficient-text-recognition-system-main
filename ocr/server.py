"""Tiny local web app (standard library only): upload an image, get the text."""
from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import SAMPLES_DIR, WEB_DIR
from .predict import recognize

MAX_UPLOAD = 20 * 1024 * 1024


def serve(model, info: dict, device: str, port: int = 8000, open_browser: bool = True) -> None:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:        # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                samples = sorted(p.name for p in SAMPLES_DIR.glob("*.png"))
                self._json({**info, "device": device, "samples": samples})
            elif path.startswith("/samples/"):
                file = SAMPLES_DIR / path.rsplit("/", 1)[-1]      # basename only
                if file.is_file():
                    ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                    self._send(200, file.read_bytes(), ctype)
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/recognize":
                return self._json({"error": "not found"}, 404)
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_UPLOAD:
                return self._json({"error": "Send one image of at most 20 MB."}, 400)
            data = self.rfile.read(size)
            try:
                with lock:
                    result = recognize(model, data, device)
            except Exception as exc:                 # unreadable / not an image
                return self._json({"error": f"Could not read that image ({exc})."}, 400)
            self._json(result)

    for candidate in range(port, port + 20):         # next free port if taken
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"No free port between {port} and {port + 19}.")
    url = f"http://127.0.0.1:{candidate}"
    print(f"\nText recognition app running at {url}   (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
