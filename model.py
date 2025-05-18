import socket
import threading
import json
import os
import re
import webbrowser
import wikipedia
from google import genai
from google.genai import types

# Custom modules
from fileOpen import open_whatsapp
from current_time import TellTime
from weather import Get_Info
from set_alarm import set_alarm
from set_reminder import set_reminder
from GenAI import GenAI_search
from greet import Greetings

# Set API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Gemini Client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="""
        You are JARVIS, a witty and concise AI assistant.
        You can handle tasks like:
        - Telling the time.
        - Setting alarms or reminders.
        - Opening websites like YouTube, Wikipedia, Google.
        - Reading the news.
        - Searching on Wikipedia or YouTube.

        If the user requests one of these, respond with:
        CALL_FUNC: {"name": "<function_name>", "arguments": {...}}

        Otherwise, answer in plain English.
        """,
        temperature=0.3,
    )
)

# Pre-compiled regex
youtube_pattern = re.compile(r'\bon youtube\b', re.IGNORECASE)
wikipedia_pattern = re.compile(r'\bon wikipedia\b', re.IGNORECASE)

# Website shortcuts
sites = {
    "youtube": "https://youtube.com",
    "wikipedia": "https://wikipedia.com",
    "google": "https://google.com",
    "spotify": "https://open.spotify.com",
    "whatsapp": ""
}

# Gemini wrapper
def gemini_chat(prompt):
    print(f"[JARVIS Input]: {prompt}")
    try:
        response_stream = chat.send_message_stream(prompt)
        full_response = ""
        for chunk in response_stream:
            full_response += chunk.text
        print(f"[JARVIS Response]: {full_response}")
        return full_response
    except Exception as e:
        print(f"[JARVIS Error]: {e}")
        return ""

# Function dispatcher

def call_function(name, args):
    if name == "tell_time":
        return TellTime()
    if name == "get_weather":
        _, city, temp, sky, wind = Get_Info()
        return f"{city}: {temp}°C, {sky}, wind {wind} km/h"
    if name == "set_alarm":
        threading.Thread(target=set_alarm, args=(args["time"],), daemon=True).start()
        return f"Alarm set for {args['time']}."
    if name == "set_reminder":
        threading.Thread(target=set_reminder, args=(args["time"], args["task"]), daemon=True).start()
        return f"Reminder set: {args['task']} at {args['time']}"
    if name == "open_whatsapp":
        open_whatsapp()
        return "WhatsApp opened."
    if name == "web_search":
        if args.get("source") == "wikipedia":
            return wikipedia.summary(args.get("query"), sentences=2)
        else:
            return GenAI_search(args.get("query"))
    return "Sorry, I don't recognize that function."

# Fallback rule-based logic
forAi = False
def rule_based_response(speech):
    global forAi
    speech = speech.lower()

    if "hey jarvis" in speech:
        return Greetings() + "..."
    for name, url in sites.items():
        if f"open {name}" in speech:
            webbrowser.open(url)
            return f"Opening {name} sir..."
    if "what's the time" in speech:
        return "The time is, " + TellTime() + ", sir..."
    if "the weather" in speech:
        wt, city, temp, sky, wind = Get_Info()
        return f"Weather in {city} as of {wt}: {temp}, {sky}, wind {wind} km/h..."
    if "set an alarm" in speech:
        match = re.search(r'\b\d{1,2}:\d{2}\s*[ap]\.m\.', speech)
        if match:
            time = match.group(0).replace(".", "").lower()
            threading.Thread(target=set_alarm, args=(time,), daemon=True).start()
            return f"Alarm set for {time}, sir..."
        return "Could not set an alarm, sir..."
    if "remind me" in speech:
        match = re.search(r'remind me to (.+?) at (\d+:\d+ [ap]\.m\.)', speech)
        if match:
            task, time = match.groups()
            time = time.replace(".", "").lower()
            threading.Thread(target=set_reminder, args=(time, task), daemon=True).start()
            return f"Reminder set to {task} at {time}, sir..."
        return "Could not set a reminder, sir..."
    if "open app" in speech:
        open_whatsapp()
        return "Opening WhatsApp sir..."
    if "can you access this code" in speech:
        return "Accessing the code is possible. Please specify which code you would like to access..."

    if "search for" in speech or forAi:
        if youtube_pattern.search(speech):
            forAi = False
            query = youtube_pattern.sub('', speech).replace("search for", "").strip()
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return "Opening YouTube sir..."
        elif wikipedia_pattern.search(speech):
            forAi = False
            query = wikipedia_pattern.sub('', speech).replace("search for", "").strip()
            return wikipedia.summary(query, sentences=2) + "..."
        elif "on the net" in speech or forAi:
            query = speech.replace("search for", "").replace("on the net", "").strip()
            forAi = True
            return GenAI_search(query) + "..."
    if "that's it for now" in speech:
        return "I'll be glad to help you again, sir..."

    return "Didn't understand, sir..."

# AI response handler
def ai_response(user_input):
    assistant_text = gemini_chat(user_input)

    result = assistant_text  # default

    if assistant_text.startswith("CALL_FUNC:"):
        payload = assistant_text.replace("CALL_FUNC:", "", 1).strip()
        try:
            call_info = json.loads(payload)
            fn_name = call_info.get("name")
            fn_args = call_info.get("arguments", {})
            fn_result = call_function(fn_name, fn_args)

            followup_prompt = f"Function `{fn_name}` returned: {fn_result}"
            result = gemini_chat(followup_prompt)  # ← assign to result
        except json.JSONDecodeError:
            result = assistant_text

    # Ensure the result ends with '...'
    if not result.strip().endswith("..."):
        result += "..."
    return result

# Socket server
def handle_client(conn):
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            user_speech = data.decode()
            print(f"[Client]: {user_speech}")
            reply = ai_response(user_speech)
            conn.sendall(reply.encode())

def serverstart():
    HOST, PORT = '0.0.0.0', 12345
    with socket.socket() as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Listening on {HOST}:{PORT}")
        while True:
            conn, addr = server_socket.accept()
            print(f"Client connected: {addr}")
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

if __name__ == "__main__":
    serverstart()
