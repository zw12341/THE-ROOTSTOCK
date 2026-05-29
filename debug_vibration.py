"""
THE ROOTSTOCK — Vibration Sensor Debug Monitor
===============================================
Standalone diagnostic tool. Run this script (independently of rootstock.py)
to inspect raw Arduino sensor data and verify the vibration pipeline before
running the full installation.

Displays a live terminal readout showing:
  - Raw Arduino ADC value (0–1023, already 100× amplified by the sketch)
  - Clean signal (raw minus NOISE_FLOOR, floored at 0)
  - Presence score (exponentially smoothed, 0.0–1.0)
  - Spike detection events (sharp upward crossings of SPIKE_THRESHOLD)
  - Estimated poetry parameters (interval, words/line) that rootstock.py would use

Important: this script and rootstock.py cannot run at the same time —
they both open the Arduino serial port, which is exclusive on macOS.
Stop rootstock.py before running this tool.

Parameters are kept in sync with rootstock.py so debug output reflects
the real installation behavior. If you change NOISE_FLOOR or smoothing
coefficients in rootstock.py, update them here too.

Author: Yvonne Wang
"""

import glob
import time


def find_arduino_port() -> list:
    """
    Scan macOS serial device paths for connected Arduino boards.

    Output: list of device path strings (may be empty if no Arduino found)
    """
    candidates = (glob.glob('/dev/cu.usbmodem*') +
                  glob.glob('/dev/cu.usbserial*') +
                  glob.glob('/dev/tty.usbmodem*'))
    return candidates


def test_connection() -> str | None:
    """
    Check for an Arduino on any known serial port and report findings.

    If no Arduino-style port is found, falls back to listing all available
    serial ports via pyserial's list_ports to assist troubleshooting.

    Output: port path string (e.g. '/dev/cu.usbserial-10'), or None if not found.
    """
    print("=" * 45)
    print("ROOTSTOCK — Vibration Sensor Debug")
    print("=" * 45)

    ports = find_arduino_port()

    if not ports:
        print("✗ No Arduino port found")
        print("  → Check USB cable is connected")
        print("  → Close Arduino IDE Serial Monitor (releases port)")
        print("\nScanning all available serial ports:")
        try:
            import serial.tools.list_ports
            all_ports = list(serial.tools.list_ports.comports())
            if all_ports:
                for p in all_ports:
                    print(f"  {p.device} — {p.description}")
            else:
                print("  (no serial ports found)")
        except ImportError:
            print("  (install pyserial: pip install pyserial)")
        return None

    print(f"✓ Found {len(ports)} port(s):")
    for p in ports:
        print(f"  {p}")
    return ports[0]


def monitor_vibration(port: str):
    """
    Open the Arduino serial port and display a live sensor dashboard.

    Input: port — serial device path (e.g. '/dev/cu.usbserial-10')

    Signal processing (mirrors rootstock.py vibration_thread exactly):
      val       = raw Arduino output (0–1023, 100× amplified by sketch)
      val_clean = max(0, val - NOISE_FLOOR)    — dead-zone filter
      norm      = val_clean / recent_max        — adaptive normalization [0, 1]
      presence  = 0.70 * presence + 0.30 * norm — exponential smoothing

    NOISE_FLOOR (320) is set above the measured idle output (~300) so that
    sensor drift and ambient electrical noise produce a clean 0 at rest.

    Spike detection:
      A spike is counted when norm crosses SPIKE_THRESHOLD upward, with
      a SPIKE_DEBOUNCE guard to prevent the same physical event being
      counted multiple times. Spikes are marked with ★N in the display.

    Display columns:
      Raw     — raw Arduino value
      Clean   — after noise floor subtraction
      Presence — smoothed score (0.000 = still, 1.000 = intense contact)
      Bar     — ASCII amplitude visualization (25 chars wide)
      Interval — estimated seconds between poem cycles
      Words/line — estimated words per line
      Status   — ⏸ still / ▶ generating

    Press Ctrl-C to stop.
    """
    try:
        import serial
    except ImportError:
        print("✗ pyserial not installed: pip install pyserial")
        return

    # Signal parameters — keep in sync with rootstock.py
    SPIKE_THRESHOLD = 0.05   # norm level that counts as a spike onset
    SPIKE_DEBOUNCE  = 0.05   # minimum seconds between counted spikes
    INTERVAL_MAX    = 20.0
    INTERVAL_MIN    = 1.0
    WORDS_MAX       = 4
    WORDS_MIN       = 2

    NOISE_FLOOR     = 320    # idle Arduino output ≈ 300; subtract to create dead zone

    while True:
        try:
            ser = serial.Serial(port, 9600, timeout=1)
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("  → Close Arduino IDE Serial Monitor and retry")
            time.sleep(2)
            continue

        print("✓ Connected — reading continuously (Ctrl-C to stop)")
        print("  Arduino output: 100× amplified deviation from calibrated baseline\n")

        recent_max      = 350.0  # adaptive normalization ceiling
        presence        = 0.0
        prev_norm       = 0.0
        last_spike_time = 0.0
        spike_count     = 0

        print(f"  {'Raw':>6}  {'Clean':>5}  {'Pres':>6}  {'':6}  {'Signal bar':25}  Interval  Words  Status")
        print(f"  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*25}  {'─'*8}  {'─'*5}  {'─'*8}")

        try:
            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except serial.SerialException as e:
                    print(f"\n✗ Serial error: {e}")
                    break

                if not line:
                    continue

                try:
                    val = float(line)
                except ValueError:
                    continue

                # Discard concatenated serial frames (impossible for a valid 10-bit ADC reading)
                if val > 1023:
                    continue

                val_clean  = max(0.0, val - NOISE_FLOOR)
                recent_max = max(recent_max * 0.990, max(val_clean, 350.0))
                norm       = val_clean / recent_max
                presence   = min(1.0, presence * 0.70 + norm * 0.30)

                # Detect upward norm crossings as discrete spike events
                now        = time.time()
                spike_flag = ''
                if (norm > SPIKE_THRESHOLD
                        and prev_norm <= SPIKE_THRESHOLD
                        and now - last_spike_time > SPIKE_DEBOUNCE):
                    spike_count    += 1
                    last_spike_time = now
                    spike_flag      = f'★{spike_count}'
                prev_norm = norm

                # Compute what rootstock.py would choose for this presence level
                interval   = max(INTERVAL_MIN,
                                 INTERVAL_MAX - presence * (INTERVAL_MAX - INTERVAL_MIN))
                words_line = max(WORDS_MIN,
                                 int(WORDS_MAX - presence * (WORDS_MAX - WORDS_MIN)))

                bar    = '█' * int(norm * 25) + '░' * (25 - int(norm * 25))
                status = "⏸ still" if presence < 0.02 else "▶ generating"

                print(
                    f"\r  {val:>6.0f}  {val_clean:>5.0f}  {presence:>6.3f}  "
                    f"{spike_flag:<6}  {bar}  {interval:>6.1f}s  {words_line:>3}w  {status}   ",
                    end='', flush=True
                )

        except KeyboardInterrupt:
            break
        finally:
            try:
                ser.close()
            except Exception:
                pass

        print("\n\nSerial port disconnected or closed.")
        print("Reconnecting in 2 seconds... (stop with Ctrl-C)")
        time.sleep(2)

    print("\n\nStopped.")


if __name__ == "__main__":
    port = test_connection()
    if port:
        print("\nTouch or breathe on the sensor — watch the parameters respond.\n")
        monitor_vibration(port)
