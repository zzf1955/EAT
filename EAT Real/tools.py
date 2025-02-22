import socket
import json

def send_json_to_target(data, target_ip, target_port):

    try:
        json_data = json.dumps(data)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.bind(('', 0))
            client_socket.connect((target_ip, target_port))
            client_socket.sendall(json_data.encode('utf-8'))
            print(f"Data sent successfully from port {client_socket.getsockname()[1]}")
    except Exception as e:
        print(f"Failed to send data: {e}")

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    free_port = s.getsockname()[1]
    s.close()
    return free_port