import geocoder
import requests

def Get_Info():
    API_KEY = "95c817bac3d7c036463be9879f06a513"
    g = geocoder.ip('me')
    latitude, longitude = g.latlng
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        city = data['name']
        temp = data['main']['temp']
        sky = data['weather'][0]['description']
        wind_speed = data['wind']['speed']
        time = "N/A"  # OpenWeatherMap doesn't provide time directly
        return time, city, f"{temp}°C", sky, f"{wind_speed} km/h"
    else:
        return "N/A", "N/A", "N/A", "N/A", "N/A"
