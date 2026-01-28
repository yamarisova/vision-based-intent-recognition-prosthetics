import time, sys, serial
from serial.tools import list_ports 

DWELL_SEC = 1.8

def list_all_ports():
    ports = list(list_ports.comports())
    print("Знайдені порти:")
    for p in ports:
        print("  -", p.device)
    return ports

def pick_port():
    ports = list_all_ports()
    if not ports:
        sys.exit("Портів не знайдено. Підключи плату.")
    cu = [p.device for p in ports if "/cu." in p.device]
    tty = [p.device for p in ports if "/tty." in p.device]
    usb = [p.device for p in ports if ("usbmodem" in p.device.lower() or "usbserial" in p.device.lower())]
    for cand in cu:
        if "usbmodem" in cand.lower() or "usbserial" in cand.lower():
            return cand
    if cu: return cu[0]
    if usb: return usb[0]
    return ports[0].device

def drain(ser, dur=0.05):
    time.sleep(dur)
    while ser.in_waiting:
        print("RX:", ser.readline().decode(errors="ignore").strip())

def main(mode="local"):
    port = pick_port()
    print("Відкриваю порт:", port)
    ser = serial.Serial(port, 115200, timeout=0.8, write_timeout=0.8)
    try:
        time.sleep(2.0)
        ser.reset_input_buffer()
        time.sleep(0.2)
        ready = ser.readline().decode(errors="ignore").strip()
        print("RX:", ready or "(порожньо)")

        if mode == "testfw":
            ser.write(b"TEST_ON\n"); print("TX: TEST_ON")
            print("Йде цикл на платі. Натисни Ctrl+C щоб зупинити.")
            while True:
                drain(ser, 0.2)
        else:
            print("Локальний цикл GRASP↔RELEASE. Ctrl+C = вихід.")
            while True:
                ser.write(b"GRASP\n");   print("TX: GRASP");   drain(ser, 0.1); time.sleep(DWELL_SEC)
                ser.write(b"RELEASE\n"); print("TX: RELEASE"); drain(ser, 0.1); time.sleep(DWELL_SEC)

    except KeyboardInterrupt:
        print("\nЗупиняю...")
        try:
            if mode == "testfw":
                ser.write(b"TEST_OFF\n"); print("TX: TEST_OFF"); drain(ser, 0.2)
            else:
                ser.write(b"CLEAR\n");    print("TX: CLEAR");    drain(ser, 0.2)
        except Exception:
            pass
    except Exception as e:
        print("Помилка:", e)
    finally:
        ser.close()
        print("Готово.")

if __name__ == "__main__":
    #   python port_test.py
    #   python port_test.py testfw
    mode = "testfw" if (len(sys.argv) > 1 and sys.argv[1].lower().startswith("test")) else "local"
    main(mode)
