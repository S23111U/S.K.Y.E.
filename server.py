import socket
import threading

device_counter = 0


def handle_client(conn, addr, device_number):
    print(f'Connected by {addr}, Device {device_number}')
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            print(f'Received from Device {device_number}:', data.decode())
        except Exception as e:
            print(f"Error receiving data from {addr}, Device {device_number}:", str(e))
            break
    conn.close()
    print(f"Connection closed by {addr}, Device {device_number}")


def server(host='0.0.0.0', port=12345):
    global device_counter
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print("Server listening on", (host, port))

            while True:
                conn, addr = s.accept()
                device_counter += 1
                threading.Thread(target=handle_client, args=(conn, addr, device_counter)).start()
    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    server()
