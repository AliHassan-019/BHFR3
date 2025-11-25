import serial
import time

PORT = "/dev/ttyAMA0"   # Raspberry Pi 5 UART
BAUD = 115200

print("[INFO] Opening UART port", PORT)

try:
    uart = serial.Serial(
        PORT,
        BAUD,
        timeout=1,
        write_timeout=1
    )
except Exception as e:
    print("[ERROR] Could not open UART:", e)
    exit()

print("[INFO] UART opened successfully.")
print("[INFO] Starting loopback test...")
print("--------------------------------------------------")

counter = 0

while True:
    try:
        # Prepare test message
        message = f"Test-{counter}\n"
        
        # Send data
        uart.write(message.encode('utf-8'))
        print(f"[TX] Sent: {message.strip()}")

        time.sleep(0.05)

        # Read response
        received = uart.readline().decode("utf-8", errors="ignore").strip()

        if received:
            print(f"[RX] Received: {received}")
        else:
            print("[RX] No data")

        print("--------------------------------------------------")

        counter += 1
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting.")
        break

    except Exception as e:
        print("[ERROR] Runtime exception:", e)
        time.sleep(1)
