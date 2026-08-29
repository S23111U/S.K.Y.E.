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
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
        print(f"🌍 UI available at http://localhost:{HTTP_PORT}/{HTML_PATH}")
        httpd.serve_forever()


# ---------- Auto open browser ----------
def open_browser():
    webbrowser.open(f"http://localhost:{HTTP_PORT}/{HTML_PATH}")


# ---------- WebSocket <-> TCP Bridge ----------
async def handle_browser(websocket):
    print("🌐 Browser connected")
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((AI_HOST, AI_PORT))
    reader = FrameReader()

    try:
        async for message in websocket:
            print(f"[Browser]: {message}")
            await loop.sock_sendall(sock, message.encode())

            while True:
                chunk = await loop.sock_recv(sock, 4096)
                if not chunk:
                    return
                done = False
                for f in reader.feed(chunk):
                    await websocket.send(json.dumps(f))
                    if f["type"] in ("done", "error"):
                        done = True
                if done:
                    break
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
