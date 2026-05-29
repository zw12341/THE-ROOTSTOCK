"""
THE ROOTSTOCK — Generative Poetry System
=========================================
A bioart installation that translates the living genome of Arabidopsis thaliana (CCA1 gene)
into generative English poetry, driven by vibration sensed from the plant itself.

Pipeline:
  1. Arduino piezoelectric sensor reads plant vibration → serial port → vibration_thread
  2. vibration_thread computes a presence score (0.0 = still, 1.0 = intense touch)
  3. HyenaDNA model extends a CCA1 seed sequence into new DNA nucleotides
  4. Codon triplets (3-bp windows) are looked up in codon_word_mapping.json
  5. Presence score gates which gene-functional vocabulary is allowed into each line:
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

Semantic word mapping:
  Built by build_mapping.py using sentence-transformers (all-MiniLM-L6-v2).
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

# ── Configuration ─────────────────────────────────────────────────────────────

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

    Artistic intent: presence is the plant's voice. The higher the score, the
    more the installation shifts from quiet, cyclical language toward urgent,
    stress-coded vocabulary and accelerated line production.
    """
    global _presence_score
    NOISE_FLOOR = 320  # idle Arduino output ≈ 300 (100× amp of ~3 ADC deviation)
    recent_max  = 350.0

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

        last_bcast = 0.0
        try:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                try:
                    val = float(line)
                except ValueError:
                    continue

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
# Generated offline by build_mapping.py from NCBI gene sequences + sentence-transformers.

with open('codon_word_mapping.json', 'r') as f:
    mapping_table = json.load(f)

flat_mapping   = {}  # codon (str) → word (str)
codon_meta_map = {}  # codon (str) → full metadata dict

for function, codons in mapping_table.items():
    for codon, data in codons.items():
        if codon not in flat_mapping:   # first gene function wins; avoids cross-function collision
            flat_mapping[codon]   = data["word"]
            codon_meta_map[codon] = data

print(f"✓ Mapping loaded: {len(flat_mapping)} codons")

# ── Load HyenaDNA ──────────────────────────────────────────────────────────────

print("Loading HyenaDNA model...")
hyena_tokenizer = AutoTokenizer.from_pretrained(
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    trust_remote_code=True
)
hyena_model = AutoModel.from_pretrained(
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    trust_remote_code=True,
    return_dict=True
)
hyena_model.config.return_dict = True
print("✓ Model loaded")

PROJECTION_HEAD_PATH = "projection_head_finetuned.pt"

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
# If projection_head_finetuned.pt exists (produced by train_projection_head.py) those
# weights are loaded; otherwise Xavier-init random weights are used — run
# train_projection_head.py on the CCA1 FASTA to get meaningful nucleotide predictions.
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
    print("Press Enter to pause/resume · Ctrl-C to quit\n")
    while True:
        input()
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
      The projection loads a fine-tuned head from disk when available; otherwise
      it falls back to Xavier-init random weights.
      Only the last 512 characters of context are fed per step to respect
      the model's sequence length limit.

    Artistic intent:
      HyenaDNA was trained on 3,000+ genomes. Its hidden states encode
      deep biological grammar: codon usage bias, GC content patterns,
      regulatory motifs. The resulting DNA is not random — it follows
      genomic logic, making the generated poetry structurally grounded
      in actual molecular biology.
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


def sequence_to_words(sequence: str, mapping: dict, presence: float = 0.5) -> tuple:
    """
    Translate a DNA sequence into a list of English words using the codon mapping,
    with presence-weighted probabilistic filtering per gene function.

    Input:
      sequence — raw DNA string (will be uppercased and split into codons)
      mapping  — flat_mapping dict: codon → word
      presence — current vibration score in [0.0, 1.0]

    Output: (words: list[str], codons: list[str])
      Parallel lists; words[i] is the English translation of codons[i].

    Filtering logic:
      Each codon belongs to a gene functional category (circadian / photosynthesis
      / stress_response). Its inclusion probability is drawn from FUNCTION_WEIGHTS
      evaluated at the current presence level. A codon is included only if
      random.random() < that probability.

      This means the same DNA sequence produces different poetry depending on
      plant state — not by changing which DNA is generated, but by changing
      which translations are allowed to surface. The poem is shaped by the
      plant's physiology in real time.

    Artistic intent:
      Silence is as meaningful as words. At rest, stress_response codons are
      almost entirely suppressed, giving the poem a slow, cyclical character.
      During intense vibration, circadian words recede and boundary/threshold
      language erupts — as if the plant's defensive signaling becomes audible.
    """
    sequence = sequence.upper()
    codons   = [sequence[i:i+3]
                for i in range(0, len(sequence) - 2, 3)
                if len(sequence[i:i+3]) == 3]
    words, codons_out = [], []
    for codon in codons:
        if codon not in mapping:
            continue
        fn   = codon_meta_map.get(codon, {}).get("gene_function", "")
        prob = FUNCTION_WEIGHTS.get(fn, lambda p: 0.4)(presence)
        if random.random() < prob:
            words.append(mapping[codon])
            codons_out.append(codon)
    return words, codons_out


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
        #   memory=0.0 → temperature=0.5 (conservative, close to CCA1 statistics)
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

        new_words, new_codons = sequence_to_words(new_dna, flat_mapping, presence)
        if not new_words:
            continue

        # Broadcast per-codon metadata for OSC and browser UI
        for codon in new_codons:
            meta = codon_meta_map.get(codon, {})
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
        for word, codon in zip(new_words, new_codons):
            line_buffer.append(word)
            codon_buffer.append(codon)
            word_count += 1
            current_word_count = word_count

            if len(line_buffer) >= words_per_line:
                line = " ".join(line_buffer)
                print(line)

                # Compute the dominant gene function for this line so the browser
                # can color the codon tag to match the actual codons displayed.
                fns = [codon_meta_map.get(c, {}).get("gene_function", "")
                       for c in codon_buffer]
                dominant_fn = max(set(fns), key=fns.count) if fns else ""

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

        time.sleep(cycle_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\ngrowth interrupted.")
