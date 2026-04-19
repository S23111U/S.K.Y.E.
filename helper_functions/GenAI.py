import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
chatHistory = []

def GenAI_search(prompt):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    chatHistory.append({'role': 'user', 'parts': [prompt]})
    response = model.generate_content(chatHistory)
    chatHistory.append({'role': 'model', 'parts': [response.text]})
    return response.text
