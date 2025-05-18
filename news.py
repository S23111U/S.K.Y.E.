import requests
import speech_recognition as sr
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio, language="en-IN")
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Could you please repeat?")
            return listen()
        except sr.RequestError:
            speak("Sorry, my speech service is down. Please try again later.")
            return None

def latestnews(query):
    API_KEY = 'pub_5670735eeb1d2e0dd7dd2a770cb429eb37f8f'
    url = f'https://newsdata.io/api/1/latest?apikey={API_KEY}&q={query}'
    response = requests.get(url)

    if response.status_code != 200:
        speak("Failed to retrieve news")
        return
    
    news = response.json()
    articles = news.get('results', [])

    speak("Top 5 News Titles:")
    for i, article in enumerate(articles[:5]):
        title = article.get('title')
        speak(f"{i+1}. {title}")

    speak("Do you wish to know more about any of these news?")
    knowmorechoice = listen().lower()

    if knowmorechoice == "yes":
        speak("Enter the number of the title you want to know more about:")
        choice = int(listen())
        if 1 <= choice <= 5:
            selected_article = articles[choice-1]
            title = selected_article.get('title')
            description = selected_article.get('description')
            speak(f"Title: {title}\nDescription: {description}")
        else:
            speak("Invalid choice")

def callingfetchnews():
    while True:
        speak("For which topic do you want to listen to the news, sir?")
        query = listen().lower()
        if query:
            latestnews(query)
        speak("Shall I fetch more news for you?")
        continue_choice = listen().lower()
        if continue_choice != 'yes':
            speak("That's it for news now.")
            break

if __name__ == "__main__":
    callingfetchnews()
