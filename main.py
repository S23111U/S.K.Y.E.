# import os
import socket
import webbrowser
import re
import wikipedia
from fileOpen import open_whatsapp
import threading

# For web search
from GenAI import GenAI_search

# For current time
from current_time import TellTime

# For greetings
from greet import Greetings

# For weather
from weather import Get_Info

# For alarm
from set_alarm import set_alarm

# For reminder
from set_reminder import set_reminder

# Variable List
HOST = '192.168.1.19'
PORT = 12345
youtube_pattern = re.compile(r'\bon youtube\b', re.IGNORECASE)
wikipedia_pattern = re.compile(r'\bon wikipedia\b', re.IGNORECASE)
sites = [["youtube", "https://youtube.com"], ["wikipedia", "https://wikipedia.com"], ["google", "https://google.com"],
         ["spotify", "https://open.spotify.com"], ["whatsapp", ""]]
weather_time, city, temp, sky, pos = Get_Info()
forAi = False


def response_condition(speech):
    global forAi
    # Entry-------------------------------------------
    if f"Hey JARVIS".lower() in speech.lower():
        return Greetings() + "..."
    # -------------------------------------------

    # Feature 1: Opening sites
    for site in sites:
        if f"Open {site[0]}".lower() in speech.lower():
            webbrowser.open(site[1])
            return f"Opening {site[0]} sir" + "..."

    # Feature 2: Tell the current time
    if f"what's the time".lower() in speech.lower():
        return "The time is, " + TellTime() + ", sir" + "..."

    # Feature 3: Current weather conditions
    if f"the weather".lower() in speech.lower():
        return (f"The weather conditions in {city}  as of {weather_time} are, temperature is {temp}, sky is {sky} "
                f"and wind is {pos} kilometer per hour" + "...")

    # Feature 4: Set an alarm
    if f"set an alarm".lower() in speech.lower():
        pattern = r'\b\d{1,2}:\d{2}\s*[ap]\.m\.'
        match = re.search(pattern, speech, re.IGNORECASE)
        if match:
            time = match.group(0)
            time = time.replace(".", "").lower()
            alarm_thread = threading.Thread(target=set_alarm, args=(time,))
            alarm_thread.start()
            return f"Alarm set for {time}, sir" + "..."
        else:
            return "Could not set an alarm, sir" + "..."

    # Feature 5: Set a reminder
    if f"remind me".lower() in speech.lower():
        pattern = r'remind me to (.+?) at (\d+:\d+ [ap]\.m\.)'
        match = re.search(pattern, speech, re.IGNORECASE)
        if match:
            task = match.group(1)
            time = match.group(2)
            time = time.replace(".", "").lower()
            reminder_thread = threading.Thread(target=set_reminder, args=(time, task,))
            reminder_thread.start()
            return f"Reminder set to {task} at {time}, sir" + "..."
        else:
            return "Could not set a reminder, sir" + "..."

    # Feature 6: Read out the news
    if f"news".lower() in speech.lower():
        pass

    if f"open app".lower() in speech.lower():
        # speech = speech.replace("open ", "")
        open_whatsapp()
        return "Opening whatsapp sir" + "..."

    # Feature No.: Searching on the internet
    if (f"search for".lower() in speech.lower()) or forAi:
        # On YOUTUBE
        if youtube_pattern.search(speech):
            forAi = False
            speech = youtube_pattern.sub('', speech)
            speech = speech.replace("search for", "")
            youtube_link = "https://www.youtube.com/results?search_query=" + speech.strip()
            webbrowser.open(youtube_link)
            return "Opening sir" + "..."

        # On Wikipedia
        elif wikipedia_pattern.search(speech):
            forAi = False
            speech = wikipedia_pattern.sub('', speech)
            speech = speech.replace("search for", "")
            result = wikipedia.summary(speech, sentences=2)
            return result + "..."

        # On the net
        elif f"on the net".lower() in speech.lower() or forAi:
            speech = speech.replace("search for", "")
            query = speech.replace("on the net", "")
            GenAI_result = GenAI_search(query)
            forAi = True
            return GenAI_result + "..."

    else:
        forAi = False

    # Exit-------------------------------------------
    if f"That's it for now".lower() in speech.lower():
        return "I'll be glad to help you again, sir" + "..."
    # -------------------------------------------

    return "Didn't understand, sir" + "..."


# Connection Acceptance
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))

    server_socket.listen()
    print(f"Server is listening for connection on {HOST, PORT}")

    conn, addr = server_socket.accept()
    with conn:
        print(f"Connected by {addr}")

        while True:
            data = conn.recv(1024)
            if not data:
                break

            # Receive data from client
            print(f"Received from client: {data.decode()}")
            data_received = data.decode()

            # Send specific result based on the request
            response = response_condition(data_received)
            conn.sendall(response.encode())
