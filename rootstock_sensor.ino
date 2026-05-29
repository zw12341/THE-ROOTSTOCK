/**
 * THE ROOTSTOCK — Vibration Sensor Sketch
 * ========================================
 * Reads a piezoelectric vibration sensor on analog pin A0 and outputs
 * the amplified deviation from a calibrated baseline over Serial at 25 Hz.
 *
 * The output is a single integer per line (0–1023), representing how much
 * the sensor has deviated from its resting state, amplified 100×.
 * At rest the output is near 0; physical contact or air movement produces
 * values well above 0, which Python reads as the plant's "presence."
 *
 * Signal chain:
 *   raw ADC (0–1023) → subtract baseline → ×100 amplify → constrain → Serial.println
 *
 * Baseline:
 *   Computed as an average of BASELINE_SAMPLES readings taken at startup
 *   while the sensor is at rest. This removes the DC offset inherent to
 *   the piezo circuit. The baseline is fixed after boot — for long sessions,
 *   restart the Arduino if baseline drift causes elevated idle output.
 *
 * Amplification:
 *   The raw deviation from baseline is typically 1–5 ADC units for gentle
 *   touch or airflow, which would be invisible at 1× scale. ×100 brings
 *   those subtle signals into the measurable range (100–500), while the
 *   constrain() call prevents arithmetic overflow.
 *
 * Python-side noise floor:
 *   The idle output with 100× amplification is ~300 (due to unavoidable
 *   sensor noise). rootstock.py subtracts NOISE_FLOOR = 320 before
 *   computing the normalized presence score, creating a dead zone
 *   that isolates intentional touch from ambient electrical noise.
 *
 * Hardware:
 *   Sensor: Yurobot piezoelectric vibration sensor module
 *   Board:  Arduino Uno (or compatible)
 *   Pin:    A0 (analog input)
 *   Baud:   9600
 *
 * Author: Yvonne Wang
 */

const int   PIN              = A0;
const int   BASELINE_SAMPLES = 100;   // Number of samples averaged at boot for baseline
const float AMPLIFY          = 100.0; // Amplification factor applied to ADC deviation

int baseline = 0;

void setup() {
  Serial.begin(9600);

  /**
   * Baseline calibration:
   * Average BASELINE_SAMPLES readings taken 10 ms apart.
   * The sensor should be undisturbed during this 1-second window.
   * The result removes the DC offset so subsequent readings represent
   * pure deviation from rest.
   */
  long sum = 0;
  for (int i = 0; i < BASELINE_SAMPLES; i++) {
    sum += analogRead(PIN);
    delay(10);
  }
  baseline = sum / BASELINE_SAMPLES;
}

void loop() {
  int raw   = analogRead(PIN);

  /**
   * Compute absolute deviation from baseline, amplify, and clamp to [0, 1023].
   *
   * abs(raw - baseline): how far the sensor has moved from its resting position.
   *   Positive for both compression and release of the piezo crystal.
   * × AMPLIFY: scales small mechanical events (wind, light touch) into a range
   *   that Python's noise-floor filter can distinguish from true signal.
   * constrain(..., 0, 1023): prevents values from exceeding the ADC integer range,
   *   which would otherwise corrupt Python's float parsing.
   */
  int delta     = abs(raw - baseline);
  int amplified = constrain((int)(delta * AMPLIFY), 0, 1023);

  Serial.println(amplified);
  delay(40); // 25 Hz output rate
}
