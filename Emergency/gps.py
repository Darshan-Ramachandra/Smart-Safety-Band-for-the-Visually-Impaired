import socket
import os

# Replace with your phone's IP and Port from the app
PHONE_IP = os.getenv("PHONE_IP", "100.66.31.220")
PORT = int(os.getenv("PHONE_PORT", "8080"))

try:
    print(f"Connecting to phone at {PHONE_IP}:{PORT}...")
    # Create a TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((PHONE_IP, PORT))
    print("Connected! Waiting for GPS data...\n")

    while True:
        # Receive up to 1024 bytes of data
        data = client_socket.recv(1024)
        if not data:
            print("Connection closed by the phone.")
            break
        
        # Decode the raw NMEA text and print it
        print(data.decode('utf-8', errors='ignore'), end='')

except KeyboardInterrupt:
    print("\nStopping receiver.")
except Exception as e:
    print(f"Error: {e}")
finally:
    client_socket.close()
