from datetime import datetime

import geocoder
import requests


def tell_time() -> str:
    """Returns the current local time, e.g. '4:07 PM'."""
    now = datetime.now()
    current_time = now.strftime("%I:%M %p")
    if current_time[0] == "0":
        current_time = current_time[1:]
    return current_time


def tell_date() -> str:
    """Returns today's date, e.g. 'Thursday, 24 September 2026'."""
    return datetime.now().strftime("%A, %-d %B %Y")


def get_weather(city: str = "") -> str:
    """Returns current weather. With no city it uses the user's location (from IP); pass city, e.g. "Salida, Colorado, USA", for anywhere else."""
    import os
    api_key = os.getenv("OPENWEATHER_API_KEY")
    place = None
    if city.strip():
        parts = [p.strip() for p in city.split(",") if p.strip()]
        # OpenWeather's geocoder wants "city,state,country" codes, not full names,
        # so try the whole thing, then progressively less.
        for q in (city, ",".join(parts[:2]), parts[0]):
            r = requests.get("http://api.openweathermap.org/geo/1.0/direct",
                             params={"q": q, "limit": 1, "appid": api_key}, timeout=10)
            if r.status_code == 200 and r.json():
                place = r.json()[0]
                break
        if not place:
            return f"I could not find a place called {city}."
        latitude, longitude = place["lat"], place["lon"]
        name = place["name"] + (f", {place['state']}" if place.get("state") and place["state"] != place["name"] else "")
    else:
        g = geocoder.ip("me")
        latitude, longitude = g.latlng
        name = None
    response = requests.get(
        "http://api.openweathermap.org/data/2.5/weather",
        params={"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}, timeout=10)
    if response.status_code != 200:
        return "I could not get the weather just now."
    data = response.json()
    name = name or data["name"]
    temp = round(data["main"]["temp"])
    feels = round(data["main"]["feels_like"])
    sky = data["weather"][0]["description"]
    out = f"In {name} it is {temp} degrees Celsius with {sky}"
    if abs(feels - temp) >= 3:
        out += f", feeling like {feels}"
    return out + "."
