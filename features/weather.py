import geocoder
from geopy.geocoders import Nominatim
import requests
from bs4 import BeautifulSoup


def Get_Info():
    geolocator = Nominatim(user_agent="geoapi")
    g = geocoder.ip('me')
    latitude, longitude = g.latlng
    lat = str(latitude)
    longi = str(longitude)
    location = geolocator.reverse(lat + "," + longi)
    address = location.raw['address']
    city = address.get('city', '')

    url = "https://www.google.com/search?q=" + "temperature in " + city

    html = requests.get(url).content

    soup = BeautifulSoup(html, 'html.parser')

    temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text

    string = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text

    data = string.split('\n')
    time = data[0]
    sky = data[1]

    listdiv = soup.findAll('div', attrs={'class': 'BNeawe s3v9rd AP7Wnd'})

    strd = listdiv[5].text

    pos = strd.find('Wind')

    # print("City:", city)
    # print("Co-ordinates:", latitude, longitude)
    # print("Temperature is", temp)
    # print("Time:", time)
    # print("Sky Description:", sky)
    # print("Wind:", pos)
    return time, city, temp, sky, pos
