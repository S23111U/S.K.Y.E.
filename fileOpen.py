# import os
# import subprocess
#
#
# def find_whatsapp_executable(start_path):
#     # Search for WhatsApp executable in the given directory and its subdirectories
#     for root, dirs, files in os.walk(start_path):
#         print(files)
#         for file in files:
#             if file.lower() == "whatsapp.exe":
#                 return os.path.join(root, file)
#     return None
#
#
# def open_whatsapp():
#     # Start searching from the root of the C: drive (can be adjusted based on your system)
#     start_path = "C:\\Program Files\\WindowsApps"
#     print(start_path)
#     # Find the WhatsApp executable
#     whatsapp_exe = find_whatsapp_executable(start_path)
#     print(whatsapp_exe)
#     if whatsapp_exe:
#         try:
#             password = input("password")
#             cmd = ["runas", "/user:swetu", whatsapp_exe]
#             process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                                        universal_newlines=True)
#             out, err = process.communicate(input=password + '\n')
#
#             # Check if the process terminated successfully
#             if process.returncode == 0:
#                 print("WhatsApp opened successfully.")
#             else:
#                 print(f"Failed to open WhatsApp. Error: {err}")
#         except FileNotFoundError:
#             print("WhatsApp not found. Make sure it's installed and accessible.")
#     else:
#         print("WhatsApp executable not found.")
#
#
# # 1973465821
# open_whatsapp()
import subprocess

command = 'start "whatsapp" "C:\\Users\\swetu\\OneDrive\\Documents\\WhatsApp Web.lnk"'


def open_whatsapp():
    subprocess.Popen(command, shell=True)
