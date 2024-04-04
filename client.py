import socket


def client():
    host = '192.168.1.11'
    port = 12345

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        while True:
            command = input("Enter command: ")
            if command.lower() == "exit":
                break
            s.sendall(command.encode())
            print("Command sent")


if __name__ == "__main__":
    client()
