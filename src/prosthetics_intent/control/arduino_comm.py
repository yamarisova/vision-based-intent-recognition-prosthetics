import time
from serial.tools import list_ports

class ArduinoComm:
    def __init__(self, baud=115200):
        self.ser = None
        self.baud = baud
        self.port = None

    def connect_auto(self):
        try:
            import serial
        except Exception as e:
            print("[SER] pyserial not installed:", e)
            return False

        try:
            candidates = [p.device for p in list_ports.comports()]
            prio = [d for d in candidates if ("usbmodem" in d.lower()
                                              or "usbserial" in d.lower()
                                              or d.lower().startswith("/dev/tty."))]
            ordered = prio + [d for d in candidates if d not in prio]
            print("[SER] Candidates:", ordered)
            if not ordered:
                print("[SER] No ports found.")
                return False
            self.port = ordered[0]
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2.0)
            print(f"[SER] Connected @ {self.port}")
            # optional ping
            try:
                self.ser.write(b"PING\n")
            except Exception:
                pass
            return True
        except Exception as e:
            print("[SER] Connect failed:", e)
            self.ser = None
            return False

    def send(self, msg: str):
        if not self.ser: return
        try:
            self.ser.write((msg if msg.endswith("\n") else msg+"\n").encode("utf-8"))
        except Exception as e:
            print("[SER] Write failed:", e)

    def close(self):
        if self.ser:
            try: self.ser.close()
            except: pass
            self.ser = None
