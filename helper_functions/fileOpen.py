import platform
import subprocess


def open_app(app_name):
    try:
        subprocess.run(["open", "-a", app_name], check=True)
        print(f"{app_name} opened successfully")
    except subprocess.CalledProcessError:
        print(f"Couldn't open {app_name}. Maybe it isn't installed?")
