"""
THE ROOTSTOCK — Generative Poetry System
=========================================
A bioart installation that translates the living genome of Arabidopsis thaliana (seeded
from the CCA1 circadian clock gene) into generative English poetry, driven by vibration
sensed from the plant itself.

Pipeline:
  1. Arduino piezoelectric sensor reads plant vibration → serial port → vibration_thread
  2. vibration_thread computes a presence score (0.0 = still, 1.0 = intense touch)
  3. HyenaDNA, with an A/C/G/T head fine-tuned on Arabidopsis circadian-clock genes,
     extends a CCA1 seed sequence into new DNA nucleotides
  4. Codon triplets (3-bp windows) are looked up in codon_word_mapping.json, where
     most codons carry one word per gene function (circadian / photosynthesis / stress)
  5. Presence score chooses which of those readings each codon takes, and gates it:
       - still  → circadian words dominate (sleep, return, night…)
       - active → stress_response words dominate (threshold, resist, rupture…)
  6. Completed lines are broadcast via WebSocket → browser visualization (index.html)
  7. OSC messages are also sent for optional audio / Max-MSP integration

Gene sources (NCBI Nucleotide):
  - Circadian rhythm:   NM_001035612  (CCA1, Arabidopsis thaliana)
  - Photosynthesis:     AY091856
  - Stress response:    NM_124370

DNA model:
  HyenaDNA — LongSafari/hyenadna-tiny-1k-seqlen-hf
  Nguyen et al., "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide
  Resolution," NeurIPS 2023. https://arxiv.org/abs/2306.15794

  The backbone is frozen. Only its four-way A/C/G/T output head is fine-tuned, starting
  from the pretrained rows, on a small set of Arabidopsis circadian-clock genes
  (CCA1, LHY, PRR9, PRR7, PRR5, TOC1, ELF3, LUX, GI, ZTL). ELF4 (circadian) and
  ECT2 (non-circadian) are held out as tests.
  See training/dataset.md and training/training.md.

Semantic word mapping:
  Built by tools/build_mapping.py using sentence-transformers (all-MiniLM-L6-v2).
  Stored in codon_word_mapping.json (generated offline, committed to repo).

Author: Yvonne Wang
"""

import json
import os
import time
import threading
import asyncio
import queue
import glob
import socket
import sys
import websockets
import serial
from pythonosc import udp_client
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random

from training.hyenadna_models import DEFAULT_HYENADNA_MODEL, resolve_hyenadna_revision

# ── Configuration ─────────────────────────────────────────────────────────────

# Simulate the Arduino instead of reading the serial port (see dummy_sample_source).
# Can also be enabled from the command line: python rootstock.py --dummy
DUMMY = False
if "--dummy" in sys.argv:
    DUMMY = True

OSC_IP        = "127.0.0.1"
OSC_PORT      = 9000
WS_PORT       = 8765

# The first 30 bp of the CCA1 coding sequence used as generative seed.
# HyenaDNA will extend this forward indefinitely during the installation.
SEED_SEQUENCE = "ATGGATCTCGAGAAGAGAAGAGTTTCAGAG"

# Generation cadence: linearly interpolated by presence score.
WORDS_PER_LINE_MIN = 2    # dense output at high vibration (short, urgent lines)
WORDS_PER_LINE_MAX = 4    # sparse output at rest (long, slow lines)
CYCLE_INTERVAL_MIN = 1.0  # seconds between cycles at full presence
CYCLE_INTERVAL_MAX = 20.0 # seconds between cycles at zero presence (heartbeat mode)

# Probability that a word from each gene function appears in a line,
# as a function of presence p ∈ [0.0, 1.0].
# At rest: circadian vocabulary dominates.
# During touch: stress_response vocabulary dominates.
# Photosynthesis acts as a constant neutral bridge between states.
FUNCTION_WEIGHTS = {
    "circadian":       lambda p: 1.0 - p * 0.78,   # 1.00 (still) → 0.22 (active)
    "photosynthesis":  lambda p: 0.55,              # stable at 0.55 in all states
    "stress_response": lambda p: 0.05 + p * 0.90,  # 0.05 (still) → 0.95 (active)
}

# ── Terminal colours ──────────────────────────────────────────────────────────
# Gene function colours are lighter versions of the codon tag colours in
# index.html (those are too dark on a dark terminal). Exact 24-bit colours
# where the terminal supports them, nearest 256-colour codes otherwise (e.g.
# macOS Terminal.app). Disabled when output is not a terminal or NO_COLOR is set.

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
TRUECOLOR = os.environ.get("COLORTERM", "") in ("truecolor", "24bit")

FUNCTION_COLORS = {          # (24-bit RGB, 256-colour fallback)
    "circadian":       ((0x6f, 0xbf, 0x94), 72),
    "photosynthesis":  ((0xd4, 0xa9, 0x4a), 179),
    "stress_response": ((0xd7, 0x7a, 0x7a), 174),
}


def color(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code (no-op when colours are disabled)."""
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def color_rgb(text: str, spec: tuple) -> str:
    """Colour text with a (24-bit RGB, 256-colour fallback) pair."""
    (r, g, b), fallback = spec
    return color(text, f"38;2;{r};{g};{b}" if TRUECOLOR else f"38;5;{fallback}")


def color_fn(text: str, fn: str) -> str:
    """Colour text with the browser colour of gene function fn."""
    return color_rgb(text, FUNCTION_COLORS[fn]) if fn in FUNCTION_COLORS else text


LIME       = ((0x7f, 0xff, 0x00), 118)  # survives
BRIGHT_RED = ((0xff, 0x30, 0x30), 196)  # discarded

# ── Presence score (thread-shared) ────────────────────────────────────────────

_presence_score = 0.0
_presence_lock  = threading.Lock()

# ── Plant memory (thread-shared) ───────────────────────────────────────────────
# Slow-decaying accumulator of touch history. Half-life ≈ 2 hours.
# Models the plant's cumulative stress state across the installation session.

_plant_memory = 0.0
_memory_lock  = threading.Lock()


def get_plant_memory() -> float:
    with _memory_lock:
        return _plant_memory


def plant_memory_thread():
    """
    Background thread: updates plant memory once per second.

    Accumulates presence score very slowly (×0.0001 weight) and decays
    at ×0.9999/second — half-life ≈ 2 hours. This means the plant
    'remembers' being touched for hours after contact ends, mirroring
    the timescale of real mechanosensory gene expression in Arabidopsis.

    This slow memory is used to modulate DNA generation temperature:
    a plant that has been touched more will produce more disordered sequences.
    """
    global _plant_memory
    while True:
        p = get_presence()
        with _memory_lock:
            _plant_memory = min(1.0, _plant_memory * 0.9999 + p * 0.0001)
        time.sleep(1)


def get_presence() -> float:
    """
    Return the current vibration presence score (0.0–1.0), thread-safely.

    Output: float in [0.0, 1.0]
      0.0 = sensor at rest, no touch detected
      1.0 = intense vibration / strong physical contact
    """
    with _presence_lock:
        return _presence_score


def get_dynamic_params() -> tuple:
    """
    Compute cycle_interval and words_per_line from the current presence score.

    Both values are linearly interpolated between their MIN/MAX bounds:
      - High presence → short interval, fewer words per line (urgent rhythm)
      - Low presence  → long interval, more words per line (slow, contemplative)

    Output: (cycle_interval: float, words_per_line: int)
    """
    s = get_presence()
    interval   = CYCLE_INTERVAL_MAX - s * (CYCLE_INTERVAL_MAX - CYCLE_INTERVAL_MIN)
    words_line = int(WORDS_PER_LINE_MAX - s * (WORDS_PER_LINE_MAX - WORDS_PER_LINE_MIN))
    return max(CYCLE_INTERVAL_MIN, interval), max(WORDS_PER_LINE_MIN, words_line)


# ── Arduino port discovery ─────────────────────────────────────────────────────

def find_arduino_port() -> str | None:
    """
    Scan macOS serial device paths for a connected Arduino board.

    Checks /dev/cu.usbmodem*, /dev/cu.usbserial*, /dev/tty.usbmodem*.
    Returns the first match, or None if no Arduino is found.

    Output: device path string (e.g. '/dev/cu.usbserial-10') or None
    """
    candidates = (glob.glob('/dev/cu.usbmodem*') +
                  glob.glob('/dev/cu.usbserial*') +
                  glob.glob('/dev/tty.usbmodem*'))
    return candidates[0] if candidates else None


# ── Vibration sensing thread ───────────────────────────────────────────────────

NOISE_FLOOR = 320  # idle Arduino output ≈ 300 (100× amp of ~3 ADC deviation)


def process_samples(samples):
    """
    Turn a stream of raw sensor values into the shared presence score.

    Input:  samples — iterable of raw values (0–1023), from the Arduino
            (serial_sample_source) or the simulator (dummy_sample_source).
    Output: none (updates _presence_score, broadcasts presence at ~5 Hz)

    Both sources go through the exact same signal processing, so DUMMY
    mode exercises the same dead zone, normalization and smoothing as
    the installation (see vibration_thread for the details).
    """
    global _presence_score
    recent_max = 350.0
    last_bcast = 0.0

    for val in samples:
        # Reject values > 1023: concatenated serial frames from buffer overflow
        if val > 1023:
            continue

        val_clean  = max(0.0, val - NOISE_FLOOR)
        recent_max = max(recent_max * 0.990, max(val_clean, 350.0))
        norm       = val_clean / recent_max

        with _presence_lock:
            _presence_score = min(1.0, _presence_score * 0.65 + norm * 0.60)
            p = _presence_score

        # Broadcast presence at ~5 Hz regardless of poetry generation state,
        # so the browser visualization always reflects live sensor data.
        now = time.time()
        if now - last_bcast > 0.2:
            ws_broadcast({"type": "presence", "level": round(p, 3)})
            last_bcast = now


def serial_sample_source(ser):
    """
    Yield raw float values read line by line from the Arduino serial port.
    Empty or unparsable lines are skipped; serial errors propagate to the caller.
    """
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        try:
            yield float(line)
        except ValueError:
            continue


# ── DUMMY mode: simulated Arduino ──────────────────────────────────────────────
# Each state emits raw values shaped like the sketch's output (gaussian around
# a mean, with a share of idle samples mixed in, as a real touch flickers).
# Tuned so that, after process_samples, presence settles in the three bands
# the installation reacts to:
#   still   → presence ≈ 0.00        (below the 0.02 gate: no words, silence)
#   touch   → presence ≈ 0.2 – 0.6   (mixed vocabulary, ~10 s cycles)
#   intense → presence ≈ 0.65 – 1.0  (stress_response vocabulary, fast cycles)

DUMMY_STATES = {
    #           raw mean  raw std  idle share  dwell (s)
    "still":   (300,      12,      0.00,       (15, 40)),
    "touch":   (430,      45,      0.25,       (10, 25)),
    "intense": (760,      140,     0.10,       (5, 15)),
}
DUMMY_RATE = 25  # Hz, same as the Arduino sketch

_dummy_override = None  # state name forced from the keyboard, or None for auto


def dummy_sample_source():
    """
    Yield simulated raw sensor values at DUMMY_RATE Hz, forever.

    In auto mode the simulator wanders between still / touch / intense,
    staying in each state for a random dwell time (a visitor approaching,
    touching, pressing, walking away). Typing 0 / 1 / 2 + Enter in the
    terminal forces a state; a + Enter returns to auto (see keyboard_listener).
    """
    state, until, shown = "still", 0.0, None
    while True:
        now = time.time()
        if _dummy_override:
            state = _dummy_override
        elif now >= until:
            state = random.choice([s for s in DUMMY_STATES if s != state])
            until = now + random.uniform(*DUMMY_STATES[state][3])
        if state != shown:
            mode = "forced" if _dummy_override else "auto"
            print("-" * 40)
            print(f"[dummy] {state} ({mode})")
            shown = state

        mean, std, idle, _ = DUMMY_STATES[state]
        if random.random() < idle:
            mean, std = DUMMY_STATES["still"][:2]
        yield min(1023.0, max(0.0, random.gauss(mean, std)))
        time.sleep(1 / DUMMY_RATE)


def vibration_thread():
    """
    Background daemon thread: reads the piezoelectric sensor via Arduino serial,
    computes a smoothed presence score, and broadcasts it to the browser.

    Signal processing pipeline (per sample, 25 Hz):
      raw    = Arduino ADC output (0–1023), already 100× amplified by the sketch
      clean  = max(0, raw - NOISE_FLOOR)   — dead-zone filter removes idle noise
      norm   = clean / recent_max          — normalize to [0, 1] using adaptive ceiling
      score  = 0.65 * old_score + 0.60 * norm  — exponential smoothing (fast rise)

    NOISE_FLOOR is set above the measured idle ADC output (~300) so that
    environmental vibration and sensor drift do not trigger false positives.
    recent_max tracks the recent signal ceiling with slow decay (×0.990/sample),
    preventing saturation after a strong touch.

    Robustness: if the serial connection drops or an unhandled exception occurs,
    the thread closes the port and retries the full connection sequence after 1 s.
    The presence score is NOT reset to 0 on disconnect — it decays naturally
    via the 0.65 multiplier in subsequent reads once reconnected.

    DUMMY mode: no serial port is opened; dummy_sample_source feeds simulated
    values through the same pipeline instead.

    Artistic intent: presence is the plant's voice. The higher the score, the
    more the installation shifts from quiet, cyclical language toward urgent,
    stress-coded vocabulary and accelerated line production.
    """
    if DUMMY:
        print("✓ DUMMY mode — simulating Arduino vibration data (no serial port)")
        print("  type 0 / 1 / 2 + Enter to force still / touch / intense, a + Enter for auto")
        process_samples(dummy_sample_source())
        return

    while True:
        port = find_arduino_port()
        if not port:
            print("✗ Arduino not found — retrying in 2 s...")
            time.sleep(2)
            continue

        try:
            ser = serial.Serial(port, 9600, timeout=1)
            print(f"✓ Arduino connected on {port} — vibration sensing active")
        except Exception as e:
            print(f"✗ Serial open failed: {e} — retrying in 2 s...")
            time.sleep(2)
            continue

        try:
            process_samples(serial_sample_source(ser))
        except Exception as e:
            print(f"✗ Serial interrupted: {e} — reconnecting...")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(1)


# ── WebSocket broadcast layer ──────────────────────────────────────────────────
# Started first so the browser can connect immediately on page load.

ws_clients = set()
ws_queue   = queue.Queue()


def ws_broadcast(payload: dict):
    """
    Enqueue a JSON payload for delivery to all connected WebSocket clients.

    Input:  payload — dict with a 'type' key and associated fields.
    Output: none (non-blocking; delivery is handled by ws_broadcaster coroutine)

    Message types used by this system:
      {"type": "line",     "text": str, "codons": str}
      {"type": "codon",    "codon": str, "word": str, "function": str, ...}
      {"type": "presence", "level": float}
      {"type": "dna",      "sequence": str}
      {"type": "cycle",    "cycle": int, "word_count": int}
      {"type": "status",   "value": "paused"|"running"}
    """
    if not ws_clients:
        return
    ws_queue.put(json.dumps(payload))


async def ws_handler(websocket):
    """
    Handle an incoming WebSocket connection from the browser.

    Registers the client in ws_clients; removes it on disconnect.
    Incoming messages from the browser are intentionally ignored —
    this is a one-way data push from Python to the visualization.

    Input:  websocket — websockets.WebSocketServerProtocol
    """
    ws_clients.add(websocket)
    print(f"✓ WebSocket client connected ({len(ws_clients)} total)")
    try:
        # Send the current runtime state immediately so the UI can
        # show the latest presence / status / cycle when it opens.
        status_value = "paused"
        try:
            status_value = "running" if running.is_set() else "paused"
        except NameError:
            status_value = "paused"

        await websocket.send(json.dumps({
            "type": "status",
            "value": status_value
        }))
        await websocket.send(json.dumps({
            "type": "presence",
            "level": round(get_presence(), 3)
        }))
        await websocket.send(json.dumps({
            "type": "cycle",
            "cycle": current_cycle if "current_cycle" in globals() else 0,
            "word_count": current_word_count if "current_word_count" in globals() else 0
        }))
        await websocket.send(json.dumps({
            "type": "dna",
            "sequence": current_sequence if "current_sequence" in globals() else SEED_SEQUENCE
        }))

        async for _ in websocket:
            pass
    finally:
        ws_clients.discard(websocket)
        print(f"✗ WebSocket client disconnected ({len(ws_clients)} remaining)")


async def ws_broadcaster():
    """
    Drain the ws_queue every 50 ms and fan out all pending messages
    to every connected client concurrently (asyncio.gather).

    Errors from individual client sends are suppressed via return_exceptions=True
    so a single broken connection does not interrupt delivery to others.
    """
    while True:
        msgs = []
        try:
            while True:
                msgs.append(ws_queue.get_nowait())
        except queue.Empty:
            pass
        if msgs and ws_clients:
            results = await asyncio.gather(
                *[c.send(m) for c in list(ws_clients) for m in msgs],
                return_exceptions=True
            )
            for result in results:
                if isinstance(result, Exception):
                    print(f"✗ WebSocket send failed: {result}")
        await asyncio.sleep(0.05)


async def ws_main():
    """
    Start the WebSocket server and run the broadcaster loop indefinitely.
    ping_interval=None disables the automatic keep-alive ping that can
    prematurely close long-lived installation connections.
    """
    async with websockets.serve(
        ws_handler, "127.0.0.1", WS_PORT,
        reuse_address=True,
        ping_interval=None,
    ):
        print(f"✓ WebSocket server ready → ws://127.0.0.1:{WS_PORT}")
        await ws_broadcaster()


def start_ws_server():
    """
    Entry point for the WebSocket daemon thread.
    Creates a dedicated asyncio event loop so the server does not
    interfere with the main thread's synchronous poetry loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(ws_main())
    except OSError as e:
        print(f"✗ WebSocket failed (port {WS_PORT} in use?): {e}")
        print(f"  fix: kill $(lsof -ti :{WS_PORT}) and restart")


running = threading.Event()
running.set()
current_cycle = 0
current_word_count = 0
current_sequence = SEED_SEQUENCE

def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False

if not is_port_available('127.0.0.1', WS_PORT):
    print(f"✗ WebSocket port {WS_PORT} is already in use."
          "\n  fix: stop the existing process using that port or change WS_PORT.")
    sys.exit(1)

threading.Thread(target=start_ws_server, daemon=True).start()
time.sleep(0.3)  # Allow server to bind before first broadcast

# ── Load codon→word mapping ────────────────────────────────────────────────────
# Generated offline by tools/build_mapping.py from NCBI gene sequences + sentence-transformers.

with open('codon_word_mapping.json', 'r') as f:
    mapping_table = json.load(f)

# A codon can carry a word in several gene functions (e.g. ATG is a circadian
# word and a stress_response word). All readings are kept; sequence_to_words
# picks one per occurrence according to the current presence score.
codon_candidates = {}  # codon (str) → list of metadata dicts, one per gene function

for function, codons in mapping_table.items():
    for codon, data in codons.items():
        codon_candidates.setdefault(codon, []).append(data)

print(f"✓ Mapping loaded: {len(codon_candidates)} codons, "
      + ", ".join(f"{fn} {len(cs)}" for fn, cs in mapping_table.items()))

# ── Load HyenaDNA ──────────────────────────────────────────────────────────────

print("Loading HyenaDNA model...")
HYENADNA_MODEL_REVISION = resolve_hyenadna_revision(DEFAULT_HYENADNA_MODEL)
hyena_tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_HYENADNA_MODEL,
    revision=HYENADNA_MODEL_REVISION,
    trust_remote_code=True
)
hyena_model = AutoModel.from_pretrained(
    DEFAULT_HYENADNA_MODEL,
    revision=HYENADNA_MODEL_REVISION,
    trust_remote_code=True,
    return_dict=True
)
hyena_model.config.return_dict = True
print("✓ Model loaded")

PROJECTION_HEAD_PATH = "training/heads/projection_head_hyenadna_tiny-1k.pt"

def load_projection_head(hidden_size: int) -> torch.nn.Linear:
    proj = torch.nn.Linear(hidden_size, 4, bias=False)
    if os.path.exists(PROJECTION_HEAD_PATH):
        try:
            proj.load_state_dict(torch.load(PROJECTION_HEAD_PATH, map_location="cpu"))
            print(f"✓ Loaded finetuned projection head from {PROJECTION_HEAD_PATH}")
        except Exception as exc:
            torch.nn.init.xavier_uniform_(proj.weight)
            print(f"⚠ Failed to load {PROJECTION_HEAD_PATH}: {exc}")
            print("  Falling back to Xavier-init random projection head")
    else:
        torch.nn.init.xavier_uniform_(proj.weight)
        print(f"⚠ {PROJECTION_HEAD_PATH} not found; using Xavier-init random projection head")
    return proj

# Instantiate once at startup so the same weights are used for every generation call.
# If the tiny-1k head produced by training/train_projection_head.py exists in
# training/heads/, those weights are loaded: the pretrained A/C/G/T rows fine-tuned
# on the circadian-gene dataset (training/dataset.md). Otherwise Xavier-init random
# weights are used.
_hidden_size = hyena_model.config.d_model
hyena_proj   = load_projection_head(_hidden_size)

# ── OSC client ────────────────────────────────────────────────────────────────

osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
print(f"✓ OSC ready → {OSC_IP}:{OSC_PORT}")

# ── Pause / resume control ─────────────────────────────────────────────────────
# The shared `running` flag and current generation state are declared
# near the top of the module so the websocket handler can safely read
# them before any background thread starts.

def keyboard_listener():
    """
    Listen for Enter key presses on stdin to toggle the running state.
    Pausing halts DNA generation and broadcasts a status message to the browser.
    Ctrl-C exits the process entirely via KeyboardInterrupt in main().
    """
    global _dummy_override
    print("Press Enter to pause/resume · Ctrl-C to quit\n")
    dummy_keys = {"0": "still", "1": "touch", "2": "intense", "a": None}
    while True:
        key = input().strip().lower()
        if DUMMY and key in dummy_keys:
            _dummy_override = dummy_keys[key]
            continue
        if running.is_set():
            running.clear()
            print("\n⏸  Paused (press Enter to resume)")
            osc_client.send_message("/rootstock/status", "paused")
            ws_broadcast({"type": "status", "value": "paused"})
        else:
            running.set()
            print("▶  Resumed\n")
            osc_client.send_message("/rootstock/status", "running")
            ws_broadcast({"type": "status", "value": "running"})


threading.Thread(target=keyboard_listener,  daemon=True).start()
threading.Thread(target=vibration_thread,   daemon=True).start()
threading.Thread(target=plant_memory_thread, daemon=True).start()


# ── DNA generation ─────────────────────────────────────────────────────────────

def hyena_extend(sequence: str, n_new: int = 30, temperature: float = 0.9) -> str:
    """
    Extend a DNA sequence by n_new nucleotides using the HyenaDNA language model.

    Input:
      sequence    — current DNA context string (A/C/G/T characters)
      n_new       — number of new nucleotides to generate (default 30 = 10 codons)
      temperature — softmax temperature; higher = more random, lower = more deterministic

    Output: string of n_new nucleotides (e.g. "ATGCCATGA…")

    Method:
      A lightweight linear projection head (4-class: A/C/G/T) is attached to
      HyenaDNA's last hidden state and sampled via multinomial distribution.
      The projection loads a head fine-tuned on Arabidopsis circadian-clock
      genes from disk when available; otherwise it falls back to Xavier-init
      random weights.
      Only the last 512 characters of context are fed per step to respect
      the model's sequence length limit.

    Artistic intent:
      HyenaDNA was trained on 3,000+ genomes. Its hidden states encode
      deep biological grammar: codon usage bias, GC content patterns,
      regulatory motifs. The resulting DNA is not random — it follows
      genomic logic, making the generated poetry structurally grounded
      in actual molecular biology. The fine-tuned head then nudges its
      next-base choices toward the composition of the plant's own clock genes.
    """
    nucleotides = 'ACGT'
    generated   = ''
    for _ in range(n_new):
        context   = (sequence + generated)[-512:]
        input_ids = hyena_tokenizer(context, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            outputs = hyena_model(input_ids, return_dict=True)
            hidden  = outputs.last_hidden_state[0, -1, :]
            logits  = hyena_proj(hidden) / temperature
            probs   = F.softmax(logits, dim=-1)
            idx     = int(torch.multinomial(probs, 1).item())
        generated += nucleotides[idx]
    return generated


def sequence_to_words(sequence: str, candidates: dict, presence: float = 0.5) -> tuple:
    """
    Translate a DNA sequence into a list of English words using the codon mapping,
    choosing each codon's gene function according to presence, then gating it.

    Input:
      sequence   — raw DNA string (will be uppercased and split into codons)
      candidates — codon_candidates dict: codon → list of metadata dicts
                   (one per gene function in which the codon carries a word)
      presence   — current vibration score in [0.0, 1.0]

    Output: (words: list[str], codons: list[str], metas: list[dict])
      Parallel lists; words[i] is the English translation of codons[i],
      metas[i] the metadata of the reading chosen (word, gene_function, scores).

    Selection logic, per codon:
      1. Choose: the codon is read in one of its gene functions, drawn with
         probability proportional to FUNCTION_WEIGHTS[fn](presence). Most codons
         carry a word in all three functions, so the same codon reads as a
         circadian word at rest and as a stress_response word under touch.
      2. Gate: the chosen word surfaces only if random.random() is below that
         same weight; otherwise the codon stays silent.

      With the current mapping and weights, the words that surface are about
      86% circadian / 14% photosynthesis / 0% stress_response at rest (p = 0),
      53% / 21% / 26% at a touch (p ≈ 0.45), and 4% / 15% / 81% at full presence.
      The same DNA sequence produces different poetry depending on plant state:
      not by changing which DNA is generated, but by changing which readings of
      it are chosen and allowed to surface.

    Artistic intent:
      Silence is as meaningful as words. At rest, stress_response readings are
      almost entirely suppressed, giving the poem a slow, cyclical character.
      During intense vibration, circadian words recede and boundary/threshold
      language erupts — as if the plant's defensive signaling becomes audible.
    """
    sequence = sequence.upper()
    codons   = [sequence[i:i+3]
                for i in range(0, len(sequence) - 2, 3)
                if len(sequence[i:i+3]) == 3]
    words, codons_out, metas = [], [], []
    for codon in codons:
        readings = candidates.get(codon)
        if not readings:
            print(f"  {codon}  no mapping{'':<42} -> {color_rgb('✗', BRIGHT_RED)}")
            continue
        weights = [FUNCTION_WEIGHTS.get(m.get("gene_function", ""), lambda p: 0.4)(presence)
                   for m in readings]
        i = random.choices(range(len(readings)), weights=weights)[0]
        fn    = readings[i].get("gene_function", "")
        pick  = weights[i] / sum(weights)  # chance this reading was the one chosen
        roll  = random.random()
        keep  = roll < weights[i]
        # Pad before colouring: ANSI codes would otherwise count towards the width.
        label = f"{fn:<15} {f'({pick:.0%})':>6}"  # percentages right-aligned in one column
        print(f"  {codon}  {color_fn(f'{label:<22}', fn)} {readings[i]['word']:<12} "
              f"roll {roll:.2f} {'<' if keep else '≥'} {weights[i]:.2f} "
              f"-> {color_rgb('✓', LIME) if keep else color_rgb('✗', BRIGHT_RED)}")
        if keep:
            words.append(readings[i]["word"])
            codons_out.append(codon)
            metas.append(readings[i])
    print()
    return words, codons_out, metas


# ── Main generation loop ───────────────────────────────────────────────────────

def main():
    """
    Core poetry generation loop. Runs synchronously on the main thread.

    Each iteration:
      1. Block if paused (running.wait())
      2. Check presence threshold — skip cycle if sensor is at rest (< 0.02)
      3. Generate 30 new DNA nucleotides with HyenaDNA
      4. Translate to words via sequence_to_words (presence-weighted)
      5. Broadcast each codon's metadata via OSC + WebSocket
      6. Accumulate words into line_buffer; flush a line when words_per_line is reached
      7. Broadcast the completed line, then sleep for cycle_interval

    The cycle_interval and words_per_line both adapt to presence in real time,
    creating a feedback loop: the plant's touch directly controls both the
    speed and the vocabulary of its own poem.
    """
    global current_cycle, current_word_count, current_sequence
    current_sequence = SEED_SEQUENCE
    word_count       = 0
    line_buffer      = []
    codon_buffer     = []
    fn_buffer        = []  # gene function chosen for each word in line_buffer
    cycle            = 0

    print("\n" + "=" * 40)
    print("THE ROOTSTOCK")
    print("Arabidopsis thaliana · CCA1")
    print("=" * 40 + "\n")

    while True:
        running.wait()

        presence = get_presence()
        if presence < 0.02:
            time.sleep(0.3)
            continue

        # Fast timescale: presence controls vocabulary style (word filtering).
        # Slow timescale: plant_memory controls DNA generation temperature.
        #   memory=0.0 → temperature=0.5 (conservative, close to circadian-gene statistics)
        #   memory=1.0 → temperature=1.5 (disordered, stress-state DNA)
        memory      = get_plant_memory()
        temperature = 0.5 + memory * 1.0
        ws_broadcast({"type": "memory", "level": round(memory, 4)})

        # Wrap DNA generation so model errors (GPU OOM, etc.) do not kill the loop.
        try:
            new_dna = hyena_extend(current_sequence, n_new=30, temperature=temperature)
        except Exception as e:
            print(f"✗ HyenaDNA inference failed: {e} — skipping cycle")
            time.sleep(1.0)
            continue

        current_sequence = (current_sequence + new_dna)[-512:]
        cycle           += 1
        current_cycle    = cycle

        print(f"[new dna] {new_dna}\n")

        new_words, new_codons, new_metas = sequence_to_words(new_dna, codon_candidates, presence)
        if not new_words:
            print("!! No new words found, continuing... !!")
            continue

        # Broadcast per-codon metadata for OSC and browser UI
        for codon, meta in zip(new_codons, new_metas):
            osc_client.send_message("/rootstock/codon",          codon)
            osc_client.send_message("/rootstock/codon/word",     meta.get("word", ""))
            osc_client.send_message("/rootstock/codon/freq",     float(meta.get("word_frequency", 0)))
            osc_client.send_message("/rootstock/codon/semantic", float(meta.get("semantic_score", 0)))
            osc_client.send_message("/rootstock/codon/function", meta.get("gene_function", ""))
            ws_broadcast({
                "type":     "codon",
                "codon":    codon,
                "word":     meta.get("word", ""),
                "freq":     float(meta.get("word_frequency", 0)),
                "semantic": float(meta.get("semantic_score", 0)),
                "function": meta.get("gene_function", ""),
            })

        cycle_interval, words_per_line = get_dynamic_params()
        ws_broadcast({"type": "presence", "level": round(presence, 3)})
        osc_client.send_message("/rootstock/presence", float(presence))

        # Accumulate words into lines; emit a line when the buffer is full.
        # word_count is a simple counter — avoids unbounded list growth over long sessions.
        for word, codon, meta in zip(new_words, new_codons, new_metas):
            line_buffer.append(word)
            codon_buffer.append(codon)
            fn_buffer.append(meta.get("gene_function", ""))
            word_count += 1
            current_word_count = word_count

            if len(line_buffer) >= words_per_line:
                line = " ".join(line_buffer)

                # Compute the dominant gene function for this line so the browser
                # can color the codon tag to match the actual codons displayed.
                dominant_fn = max(set(fn_buffer), key=fn_buffer.count) if fn_buffer else ""

                # Codon, word and source gene function stacked in columns,
                # each column as wide as the longest of the three.
                rows   = (codon_buffer, line_buffer, [f"[{fn}]" for fn in fn_buffer])
                widths = [max(map(len, col)) for col in zip(*rows)]
                for row in rows[:2]:
                    print("  " + "  ".join(s.ljust(n) for s, n in zip(row, widths)).rstrip())
                print("  " + "  ".join(color_fn(s.ljust(n), fn)
                                       for s, n, fn in zip(rows[2], widths, fn_buffer)).rstrip())
                print()

                osc_client.send_message("/rootstock/line",       line)
                osc_client.send_message("/rootstock/word_count", word_count)
                osc_client.send_message("/rootstock/dna",        new_dna)
                osc_client.send_message("/rootstock/cycle",      cycle)

                ws_broadcast({"type":     "line",
                              "text":     line,
                              "codons":   " · ".join(codon_buffer),
                              "function": dominant_fn})
                ws_broadcast({"type": "dna",   "sequence": new_dna})
                ws_broadcast({"type": "cycle", "cycle": cycle, "word_count": word_count})

                line_buffer  = []
                codon_buffer = []
                fn_buffer    = []

        time.sleep(cycle_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\ngrowth interrupted.")
