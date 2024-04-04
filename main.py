import speech_recognition as sr
import win32com.client
import webbrowser
import google.generativeai as genai
from dotenv import load_dotenv
import os
import socket
import threading

load_dotenv()

speaker = win32com.client.Dispatch("SAPI.SpVoice")
device_counter = 0


def say(text):
    speaker.Speak(text)


def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            speechinput = r.recognize_google(audio, language="en-in")
            print(f"User input: {speechinput}")
            return speechinput
        except Exception as e:
            print(e)
            return "Couldn't Understand Sir. Can you repeat again?"


# 1: Make use of llama2
# Used for the llama2 model downloaded on local machine
# import subprocess
# def LLM_search(prompt):
#     ollama_path = r"C:\Users\swetu\AppData\Local\Programs\Ollama\ollama.exe"
#     process = subprocess.Popen([ollama_path, 'run', 'llama2'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                                stderr=subprocess.PIPE, universal_newlines=True)
#     process.stdin.write(prompt + '\n')
#     process.stdin.flush()
#     output, _ = process.communicate()
#     return output.strip()

# 2: Make use of GenAI
def GenAI_search(prompt):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text


def handle_client(conn, addr, device_number):
    print(f'Connected by {addr}, Device {device_number}')
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            print(f'Received from Device {device_number}:', data.decode())
            say(f"Data received from Device {device_number}: {data.decode()}")

        except Exception as e:
            print(f"Error receiving data from {addr}, Device {device_number}:", str(e))
            break
    conn.close()
    print(f"Connection closed by {addr}, Device {device_number}")


# 3: Making connections
def server(host='0.0.0.0', port=12345):
    global device_counter
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print("Server listening on", (host, port))

            while True:
                conn, addr = s.accept()
                device_counter += 1
                threading.Thread(target=handle_client, args=(conn, addr, device_counter)).start()
    except Exception as e:
        print("Error:", str(e))


if __name__ == '__main__':
    server_thread = threading.Thread(target=server)
    server_thread.start()

    say("Hi I'm JARVIS, How may I help you sir?")

    while True:
        print("Listening...")
        speech = command()

        # todo: Opening the sites
        sites = [["youtube", "https://youtube.com"], ["wikipedia", "https://wikipedia.com"],
                 ["google", "https://google.com"]]

        for site in sites:
            if f"Open {site[0]}".lower() in speech.lower():
                say(f"Opening {site[0]} sir...")
                webbrowser.open(site[1])

        if f"search for".lower() in speech.lower():
            query_index = speech.lower().index("search for") + len("search for")
            query = speech[query_index:].strip()
            say(f"Searching for {query}, please wait sir...")
            GenAI_result = GenAI_search(query)
            say(GenAI_result)

        if f"That's it for now".lower() in speech.lower():
            say("I'll be glad to help you again...")
            exit()
