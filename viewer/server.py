"""Static server for the WebXR viewer + /telemetry sink (fps reports) + /shot sink (canvas PNG).
Usage: python server.py [port]   (serves this directory)"""
import http.server, json, os, sys, time, base64
from urllib.parse import urlparse, parse_qs

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path.startswith("/shot"):
            try:
                name = parse_qs(urlparse(self.path).query).get("name", ["shot"])[0]
                data = body.decode("utf-8", "ignore")
                if "," in data:
                    data = data.split(",", 1)[1]
                with open(f"shot_{name}.png", "wb") as f:
                    f.write(base64.b64decode(data))
            except Exception as e:
                print("shot error:", e)
            self.send_response(204); self.end_headers()
        elif self.path == "/telemetry":
            try:
                rec = json.loads(body)
                rec["ts"] = time.strftime("%H:%M:%S")
                with open("telemetry.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
            self.send_response(204); self.end_headers()
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print(f"serving {HERE} on 0.0.0.0:{PORT}")
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
