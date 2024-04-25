import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


def GenAI_search(prompt):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    ans = model.generate_content(prompt)
    return ans.text
