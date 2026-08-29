import os
import socket
import sys

import pyttsx3
import speech_recognition as sr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from core.protocol import FrameReader

HOST = "127.0.0.1"
PORT = 12345


def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        audio = r.listen(source)
        try:
            speechinput = r.recognize_google(audio, language="en-in")
            print(f"You said: {speechinput}")
            return speechinput
        except Exception as e:
            print(e)
            # Returning a sentence here made it look like something the user
            # said; it was ingested 36 times as a "memory" about them.
            return None


def receive_reply(sock):
    reader = FrameReader()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return ""
        for f in reader.feed(chunk):
            if f["type"] == "done":
                return f["text"]
            if f["type"] == "error":
                return "Something went wrong."


def say(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))

        while True:
            print("Listening...")
            message = command()
            if not message:
                continue
            try:
                client_socket.sendall(message.encode())
            except BrokenPipeError:
                print("Connection closed by server")
                break
            response = receive_reply(client_socket)
            if not response:
                print("Connection closed by server")
                break

            print(f"Response from server:\n{response}")
            say(response)
