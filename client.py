import socket
import pyttsx3
import speech_recognition as sr

HOST = '127.0.0.1'
PORT = 12345


def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            speechinput = r.recognize_google(audio, language="en-in")
            print(f"You said: {speechinput}")
            return speechinput
        except Exception as e:
            print(e)
            return "Couldn't Understand Sir. Can you repeat again?"

def receive_all(sock):
    buffer = b''
    delimiter = b'...'
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer += chunk
        if delimiter in buffer:
            break
    return buffer

def say(text):
    pyttsx3.speak(text)

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))

        while True:
            print("Listening...")
            message = command()
            try:
                client_socket.sendall(message.encode())
            except BrokenPipeError:
                print("Connection closed by server")
                break
            response = receive_all(client_socket)

            decoded = response.decode()
            print(f"Response from server:\n{decoded}")
            say(decoded)
