import socket
import json
from template import StableDiffusionCommand

def receive_task(command_port)->StableDiffusionCommand:
    """
    Listen and wait for the next task.
    Return the task as a JSON object when received.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('', command_port))  # Bind to all interfaces
            sock.listen()
            print(f"Listening for task on port {command_port}...")
            conn, addr = sock.accept()
            with conn:
                print(f"Connected by {addr}")
                data = b""
                while True:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet
                if data:
                    cmd_json = data.decode('utf-8')
                    command = StableDiffusionCommand.from_json(cmd_json)
                    print(f"return {command}")
                    return command
    except Exception as e:
        print(f"Error receiving task: {e}")
        return None

def send_result(ip, port, file_name):
    """
    Send the image result to the master node, which is already listening on this port.
    """
    port = int(port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((ip, port))
            print(f"Sending result to {ip}:{port}")
            with open(file_name, 'rb') as f:
                data = f.read()
                sock.sendall(data)
    except Exception as e:
        print(f"Error sending result to {ip}:{port}: {e}")

def receive_result(ip, port, output_file):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((ip, int(port)))
            sock.listen(1) 
            print(f"Listening for incoming data on {ip}:{port}")

            conn, addr = sock.accept()
            with conn:
                print(f"Connected by {addr}")

                # 接收数据并写入文件
                with open(output_file, 'wb') as f:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        f.write(data)

                print(f"Data received and saved to {output_file}")
    except Exception as e:
        print(f"Error receiving data on {ip}:{port}: {e}")

