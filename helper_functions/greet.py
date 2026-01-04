import datetime


def Greetings():
    hour = int(datetime.datetime.now().hour)
    if 0 < hour < 12:
        return "Good Morning, Sir"
    elif 12 <= hour <= 15:
        return "Good Afternoon, Sir"
    elif 15 < hour <= 22:
        return "Good Evening, Sir"
    elif 22 < hour:
        return "It's quite late, Good Evening Sir"
