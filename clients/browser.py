import asyncio
import websockets
import socket
import http.server
import socketserver
import json
import sys
import threading
import webbrowser
import os

# ---- CONFIG ----
AI_HOST = "127.0.0.1"
AI_PORT = 12345
WS_PORT = 8765
HTTP_PORT = 8000

# Project root = parent of "clients" folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = "clients/browser.html"

sys.path.insert(0, BASE_DIR)
from core.protocol import FrameReader


# ---------- Serve HTML over HTTP ----------
def start_http_server():
    os.chdir(BASE_DIR)  # Serve from project root
    class handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Always serve the latest UI; a cached page hid earlier changes.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()
    with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
        print(f"🌍 UI available at http://localhost:{HTTP_PORT}/{HTML_PATH}")
        httpd.serve_forever()


# ---------- Auto open browser ----------
def open_browser():
    webbrowser.open(f"http://localhost:{HTTP_PORT}/{HTML_PATH}")


# ---------- WebSocket <-> TCP Bridge ----------
# Two independent pumps rather than one send-then-wait-for-reply loop. The
# old version only ever read from the TCP socket right after forwarding a
# browser message, so anything SKYE pushed onto that socket unprompted (a
# proactive check-in, fired from the server's own scheduler thread with no
# browser message to trigger it) would sit in the kernel receive buffer
# until the browser happened to send its next message — then get read out
# of order and get spliced into that unrelated reply. Running both
# directions concurrently means a server-initiated frame reaches the browser
# the moment it arrives, regardless of what the browser is doing.
async def _pump_ws_to_tcp(websocket, sock, loop):
    async for message in websocket:
        print(f"[Browser]: {message}")
        await loop.sock_sendall(sock, message.encode())


async def _pump_tcp_to_ws(websocket, sock, loop):
    reader = FrameReader()
    while True:
        chunk = await loop.sock_recv(sock, 4096)
        if not chunk:
            return
        for f in reader.feed(chunk):
            await websocket.send(json.dumps(f))


async def handle_browser(websocket):
    print("🌐 Browser connected")
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((AI_HOST, AI_PORT))
    # loop.sock_recv/sock_sendall require a non-blocking socket to actually
    # yield to the event loop instead of blocking the whole thread. The old
    # single-pump version got away without this — one coroutine per
    # connection meant a blocking wait was merely wasted concurrency, not a
    # deadlock. With two pumps on the same connection, a blocking sock_recv
    # in one freezes the loop and starves the other outright.
    sock.setblocking(False)

    tasks = [
        asyncio.ensure_future(_pump_ws_to_tcp(websocket, sock, loop)),
        asyncio.ensure_future(_pump_tcp_to_ws(websocket, sock, loop)),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            exc = t.exception()
            if exc:
                raise exc
    except Exception as e:
        print("Browser disconnected:", e)
    finally:
        sock.close()


async def start_ws_server():
    async with websockets.serve(handle_browser, "0.0.0.0", WS_PORT):
        print(f"🌉 WebSocket running on ws://localhost:{WS_PORT}")
        await asyncio.Future()


# ---------- MAIN ----------
if __name__ == "__main__":
    print("🚀 Starting SKYE Browser Bridge")

    threading.Thread(target=start_http_server, daemon=True).start()
    threading.Thread(target=open_browser, daemon=True).start()

    asyncio.run(start_ws_server())
