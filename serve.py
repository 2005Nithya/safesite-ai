from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import os

PORT = int(os.environ.get("PORT", 8000))
DIRECTORY = os.path.join(os.path.dirname(__file__), "public")

os.chdir(DIRECTORY)

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
print(f"Serving on http://localhost:{PORT}")
httpd.serve_forever()
