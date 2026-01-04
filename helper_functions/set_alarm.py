import time
from datetime import datetime
from plyer import notification


def convert_to_24h_format(time_str):
    return datetime.strptime(time_str, '%I:%M %p').strftime('%H:%M')


def set_alarm(alarm_time):
    alarm_time_24h = convert_to_24h_format(alarm_time)

    while True:
        current_time = time.strftime("%H:%M")
        if current_time == alarm_time_24h:
            notification.notify(
                title='Alarm',
                message='Wake up!',
                app_name='Alarm App',
                timeout=10
            )
            break
        time.sleep(60)
