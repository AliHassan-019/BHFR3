import serial
import time

# Open the main UART (/dev/serial0) at 115200 baud
ser = serial.Serial(
    port="/dev/serial0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=0.1,       # read timeout in seconds
)

print("[PI] UART opened on /dev/serial0")

counter = 0

try:
    while True:
        # ---- SEND DUMMY DATA TO ESP32 ----
        msg = f"Hello from Pi #{counter}\n"
        ser.write(msg.encode("utf-8"))
        ser.flush()
        print(f"[PI] Sent: {msg.strip()}")

        # ---- TRY TO READ ANY RESPONSE FROM ESP32 ----
        incoming = ser.readline()  # reads up to newline or timeout
        if incoming:
            try:
                text = incoming.decode("utf-8", errors="ignore").strip()
            except UnicodeDecodeError:
                text = str(incoming)
            print(f"[PI] Received from ESP32: {text}")

        counter += 1
        time.sleep(1.0)  # send every 1 second

except KeyboardInterrupt:
    print("\n[PI] Stopping UART test.")

finally:
    ser.close()
    print("[PI] UART closed.")
