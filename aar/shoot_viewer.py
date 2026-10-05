"""Capture frames from the WebXR/WebGL viewer with headless Chrome (CDP over websocket-free HTTP+WS
via chrome --remote-debugging-port). For each field file, orbit the flat camera to fixed poses and
save the canvas PNG. Used to build the client-decode comparison figure.
Usage: shoot_viewer.py --fields fieldfull.bin:full,field40.bin:ours --thetas 0.3,1.9,3.4 --t 0.5
Frames land in viewer_webxr/cap_<tag>_<k>.png"""
import argparse, base64, json, os, subprocess, sys, time, urllib.request, socket
from urllib.error import URLError

ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VD = f"{ROOT}/viewer_webxr"
ap = argparse.ArgumentParser()
ap.add_argument("--fields", default="fieldfull.bin:full,field40_orig.bin:ours")
ap.add_argument("--thetas", default="0.3,1.9,3.4")
ap.add_argument("--phi", type=float, default=0.10)
ap.add_argument("--dist", type=float, default=3.0)
ap.add_argument("--t", type=float, default=0.5)
ap.add_argument("--center", default="")            # pin the orbit centre (x,y,z) so every field uses one camera          # normalized clip time to freeze at
ap.add_argument("--port", type=int, default=8123)
ap.add_argument("--w", type=int, default=900); ap.add_argument("--h", type=int, default=1100)
a = ap.parse_args()

try:
    import websocket  # websocket-client
except ImportError:
    sys.exit("need websocket-client: pip install --no-deps websocket-client")

CH = "/usr/bin/google-chrome"
prof = f"/tmp/claude-chrome-prof-{os.getpid()}"
proc = subprocess.Popen([CH, "--headless=new", "--remote-debugging-port=9333",
                         f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
                         "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                         "--remote-allow-origins=*",
                         f"--window-size={a.w},{a.h}", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def wait_cdp():
    for _ in range(60):
        try:
            return json.loads(urllib.request.urlopen("http://127.0.0.1:9333/json/list", timeout=2).read())
        except (URLError, socket.timeout, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError("chrome CDP did not come up")
tabs = wait_cdp()
ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
ws = websocket.create_connection(ws_url, timeout=120)
_id = [0]
def cdp(method, **params):
    _id[0] += 1
    ws.send(json.dumps({"id": _id[0], "method": method, "params": params}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == _id[0]:
            if "error" in m: raise RuntimeError(f"{method}: {m['error']}")
            return m.get("result", {})
def js(expr, wait=True):
    r = cdp("Runtime.evaluate", expression=expr, awaitPromise=wait, returnByValue=True)
    if r.get("exceptionDetails"):
        raise RuntimeError(json.dumps(r["exceptionDetails"])[:400])
    return r.get("result", {}).get("value")

cdp("Page.enable"); cdp("Runtime.enable")
thetas = [float(x) for x in a.thetas.split(",")]
for spec in a.fields.split(","):
    fname, tag = spec.split(":")
    _url = f"http://localhost:{a.port}/?f={fname}&t={a.t}" + (f"&c={a.center.strip()}" if a.center else "")
    print("  url:", _url)
    cdp("Page.navigate", url=_url)
    time.sleep(2.0)
    for _ in range(120):                                   # wait for the field to finish loading
        n = js("(typeof N !== 'undefined') ? N : 0", wait=False)
        if n and int(n) > 0: break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"{fname}: viewer never reported N>0")
    print(f"[{tag}] {fname}: N={int(n)}")
    for k, th in enumerate(thetas):
        js(f"theta={th}; phi={a.phi}; dist={a.dist};", wait=False)
        js(f"try{{ TFIXED={a.t} }}catch(e){{}}", wait=False)
        # a large field renders far slower than any fixed sleep, so wait on presented frames instead
        js("new Promise(r=>{let n=0;const f=()=>{if(++n>=8)r(1);else requestAnimationFrame(f);};"
           "requestAnimationFrame(f);})")
        time.sleep(0.4)
        data = js("document.getElementById('c').toDataURL('image/png')", wait=False)
        out = f"{VD}/cap_{tag}_{k}.png"
        open(out, "wb").write(base64.b64decode(data.split(",", 1)[1]))
        print("  saved", out)
ws.close(); proc.terminate()
print("done")
