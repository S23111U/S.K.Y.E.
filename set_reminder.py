import time
from datetime import datetime
from plyer import notification


def convert_to_24h_format(time_str):
    return datetime.strptime(time_str, '%I:%M %p').strftime('%H:%M')


def set_reminder(reminder_time, message):
    reminder_time_24h = convert_to_24h_format(reminder_time)
    while True:
        current_time = time.strftime("%H:%M")
        if current_time == reminder_time_24h:
            notification.notify(
                title='Reminder',
                message=message,
                app_name='Reminder App',
                timeout=10
            )
            break
        time.sleep(60)
