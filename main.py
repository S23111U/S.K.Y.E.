import socket
import webbrowser
import re
import wikipedia

# For web search
from GenAI import GenAI_search

# For current time
from current_time import TellTime

# For greetings
from greet import Greetings

# For weather
from weather import Get_Info

HOST = '192.168.1.21'
PORT = 12345
sites = [["youtube", "https://youtube.com"], ["wikipedia", "https://wikipedia.com"], ["google", "https://google.com"],
         ["spotify", "https://open.spotify.com"]]
time, city, temp, sky, pos = Get_Info()
youtube_pattern = re.compile(r'\bon youtube\b', re.IGNORECASE)
wikipedia_pattern = re.compile(r'\bon wikipedia\b', re.IGNORECASE)


def response_condition(speech):
    # Entry
    if f"Hey JARVIS".lower() in speech.lower():
        return Greetings() + "..."

    # Feature 1: Opening sites
    for site in sites:
        if f"Open {site[0]}".lower() in speech.lower():
            webbrowser.open(site[1])
            return f"Opening {site[0]} sir" + "..."

    # Feature 2: Searching on the internet
    if f"search for".lower() in speech.lower():
        # On YOUTUBE
        if youtube_pattern.search(speech):
            speech = youtube_pattern.sub('', speech)
            speech = speech.replace("search for", "")
            youtube_link = "https://www.youtube.com/results?search_query=" + speech.strip()
            webbrowser.open(youtube_link)
            return "Opening sir" + "..."
        # On the net
        elif f"on the net".lower() in speech.lower():
            speech = speech.replace("search for", "")
            query = speech.replace("on the internet", "")
            GenAI_result = GenAI_search(query)
            return GenAI_result + "..."
        # On Wikipedia
        elif wikipedia_pattern.search(speech):
            speech = wikipedia_pattern.sub('', speech)
            speech = speech.replace("search for", "")
            result = wikipedia.summary(speech, sentences=2)
            return result + "..."

    # Feature 3: Tell the current time
    if f"what's the time".lower() in speech.lower():
        return "The time is, " + TellTime() + ", sir" + "..."

    if f"the weather".lower() in speech.lower():
        return (f"The weather conditions in {city}  as of {time} are, temperature is {temp}, sky is {sky} "
                f"and wind is {pos} kilometer per hour" + "...")

    # Exit
    if f"That's it for now".lower() in speech.lower():
        return "I'll be glad to help you again, sir" + "..."

    return "Didn't understand, sir" + "..."


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

            print(f"Received from client: {data.decode()}")
            data_received = data.decode()

            response = response_condition(data_received)
            conn.sendall(response.encode())
