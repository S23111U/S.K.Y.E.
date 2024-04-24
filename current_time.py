from datetime import datetime


def TellTime():
    now = datetime.now()
    current_time = now.strftime("%I %M %p")
    if current_time[0] == '0':
        current_time = current_time[1:]
    return current_time
