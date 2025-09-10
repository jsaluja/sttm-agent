import os, time, json
import difflib
import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, AutoModelForCTC
from rapidfuzz import fuzz
from dotenv import load_dotenv
from collections import deque, Counter
import Levenshtein
from telemetry import log
import logging

logger = logging.getLogger(__name__)

# Import our new modules
from sttm_ui_controller import activate_sttm, open_sync_panel, close_sync_panel, extract_sync_code_pin
from sttm_sync_client import STTMSyncClient
from verse_dataset import get_shabad_id_for_verse

# ====================== STTM Sync Control ===================

# STTM Sync controller instance (will be initialized after anchor is set)
sttm_sync = None

def initialize_sttm_sync():
    """Initialize the STTM sync connection after anchor is set"""
    global sttm_sync
    
    # Check if we already have a connection established (possibly by orchestrator)
    if sttm_sync is not None and sttm_sync.connected:
        print("STTM sync already initialized and connected")
        return True
    
    print("Initializing STTM sync connection...")
    
    # 1. Open the sync panel
    if not open_sync_panel():
        print("Failed to open sync panel")
        return False
    
    # 2. Extract sync code and PIN using OCR
    sync_code, pin = extract_sync_code_pin()
    
    # 3. Close the sync panel
    close_sync_panel()
    
    if not sync_code or not pin:
        print("Failed to extract sync code or PIN")
        return False
    
    # 4. Initialize sync client
    sttm_sync = STTMSyncClient()
    
    # 5. Connect to STTM
    success = sttm_sync.connect_with_code_pin(sync_code, pin)
    return success

def send_verse_to_sttm(verse_id):
    """Send verse to STTM using the sync controller"""
    global sttm_sync
    
    if sttm_sync is None or not sttm_sync.connected:
        print("❌ STTM sync not initialized or not connected")
        return False
    
    # Get the shabad ID for this verse ID
    shabad_id = get_shabad_id_for_verse(verse_id)
    
    if not shabad_id:
        print(f"No shabad ID found for verse ID: {verse_id}")
        return False
    
    # Send the verse to STTM
    return sttm_sync.send_verse(shabad_id, verse_id)

# ============= Robust Gurmukhi Matching Logic as provided ==============

GURMUKHI_MATRAS = set("ਾਿੀੁੂੇੈੋੌਂੱੰ਼")
def normalize_for_match(text):
    return "".join([c for c in text if c not in GURMUKHI_MATRAS and c != " "])

MODE = "paath"
PAATH_ALLOW_SKIP = 1
PAATH_ALLOW_REPEAT = 2

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = "jssaluja/fb-mms-1b-cleaned-jssaluja_rajinder_singh-epochs-12-test-datasets-10-20250812_232950"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
OVERLAP = 1.0
SLIDING_WORDS = 24
CONF_THRESHOLD = 72
PERSISTENCE_REQUIRED = 2

ANCHOR_STRONG_SCORE = 75
ANCHOR_WEAK_SCORE = 42
ANCHOR_MIN_FAST_AGREEMENT = 3
ANCHOR_CANDIDATE_WINDOW = 4
GENERIC_TOKENS = {"ਛੰਤ", "ਰਹਾਉ", "ਸਲੋਕੁ", "ਮਹਲਾ"}
LEADING_TRIGGER_SCORE = 55
LEADING_TRIGGER_WORDS = 4

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCTC.from_pretrained(HF_MODEL, token=HF_TOKEN)
model.to(DEVICE)
model.eval()
processor = AutoProcessor.from_pretrained(HF_MODEL, token=HF_TOKEN)

INDEX_DIR = "local_banidb"
with open(os.path.join(INDEX_DIR,"line_store.json"),"r",encoding="utf8") as f:
    LINE_STORE = json.load(f)
with open(os.path.join(INDEX_DIR,"inverted.json"),"r",encoding="utf8") as f:
    INVERTED = json.load(f)
verse_ids = list(LINE_STORE.keys())
corpus_asr = [LINE_STORE[vid]["asr_text"] for vid in verse_ids]
corpus_orig = [LINE_STORE[vid]["orig_text"] for vid in verse_ids]
corpus_tokens = [line.split() for line in corpus_asr]

audio_buffer = np.zeros(int(SAMPLE_RATE*OVERLAP), dtype=np.float32)
transcript_buffer = deque(maxlen=SLIDING_WORDS)
persistence_counter = 0
last_committed_id = None
last_commit_idx = 0

anchor_queue = deque(maxlen=ANCHOR_CANDIDATE_WINDOW)
anchor_mode = True

def ensemble_score(q, t):
    q, t = normalize_for_match(q), normalize_for_match(t)
    return 0.4*fuzz.partial_ratio(q, t) + 0.3*fuzz.token_set_ratio(q, t) + 0.3*fuzz.ratio(q, t)

def get_paath_allowed_indices(last_commit_idx, verse_count):
    indices = []
    for i in range(PAATH_ALLOW_REPEAT, 0, -1):
        idx = last_commit_idx - i
        if 0 <= idx < verse_count:
            indices.append(idx)
    for i in range(0, PAATH_ALLOW_SKIP + 2):
        idx = last_commit_idx + i
        if 0 <= idx < verse_count:
            indices.append(idx)
    return sorted(set(indices))

def select_best_fuzzy_anywhere(transcript_buffer):
    N = 10
    q = normalize_for_match("".join(list(transcript_buffer)[-N:]))
    candidates = []
    for idx, vid in enumerate(verse_ids):
        md = LINE_STORE.get(vid)
        if md:
            t = normalize_for_match(md["asr_text"])
            candidates.append((ensemble_score(q, t), idx, md))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates:
        return candidates[0]
    return 0, -1, None

def select_best_fuzzy_paath(transcript_buffer, allowed_indices):
    N = 10
    q = normalize_for_match("".join(list(transcript_buffer)[-N:]))
    candidates = []
    for idx in allowed_indices:
        vid = verse_ids[idx]
        md = LINE_STORE.get(vid)
        if md:
            t = normalize_for_match(md["asr_text"])
            candidates.append((ensemble_score(q, t), idx, md))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates:
        return candidates[0]
    return 0, -1, None

def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i]==b[j]:
                dp[i+1][j+1] = dp[i][j]+1
            else:
                dp[i+1][j+1] = max(dp[i][j], dp[i+1][j], dp[i][j+1])
    return dp[m][n]

def lcs_best(asr_tokens, corpus_tokens, orig_lines, allowed_indices):
    asr_tokens_ = [normalize_for_match(w) for w in asr_tokens]
    best_lcs, best_idx = 0, -1
    for i in allowed_indices:
        toks = [normalize_for_match(w) for w in corpus_tokens[i]]
        lcs = lcs_len(asr_tokens_, toks)
        if lcs > best_lcs:
            best_lcs = lcs
            best_idx = i
    return best_lcs, best_idx, corpus_orig[best_idx] if best_idx != -1 else ''

def seqmatcher_best(asr_tokens, corpus_asr, orig_lines, allowed_indices):
    asr_str = normalize_for_match("".join(asr_tokens))
    best_ratio, best_idx = 0, -1
    for i in allowed_indices:
        t = normalize_for_match(corpus_asr[i])
        ratio = difflib.SequenceMatcher(None, asr_str, t).ratio()
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    return best_ratio, best_idx, corpus_orig[best_idx] if best_idx != -1 else ''

def levenshtein_best(asr_tokens, corpus_asr, allowed_indices):
    asr_concat = normalize_for_match("".join(asr_tokens))
    best_dist, best_idx, best_gt = float('inf'), -1, ''
    for i in allowed_indices:
        gt_concat = normalize_for_match("".join(corpus_asr[i].split()))
        dist = Levenshtein.distance(asr_concat, gt_concat)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_gt = corpus_orig[i]
    max_len = max(len(asr_concat), len(gt_concat), 1)
    score = 1 - (best_dist / max_len)
    return score, best_idx, best_gt if best_idx != -1 else ''

def pick_strongest_paath(fuzzy, lcs, seqm, lev, asr_tokens):
    scores = [(fuzzy[0], fuzzy[1]), (lcs[0], lcs[1]), (seqm[0], seqm[1]), (lev[0], lev[1])]
    best_score, best_idx = max(scores)
    return ("ensemble", best_idx)

def maybe_commit(strong_type, idx, chunk_ms, asr_text):
    global persistence_counter, last_committed_id, last_commit_idx
    vid = verse_ids[idx]
    candidate = LINE_STORE[vid]
    log("scored", chunk_ms=chunk_ms, asr_text=asr_text,
        best_verse_id=vid, best_score='auto', committed=False)
    if last_committed_id != vid:
        persistence_counter += 1
        if persistence_counter >= PERSISTENCE_REQUIRED:
            # Send verse to STTM using the sync controller
            success = send_verse_to_sttm(vid)
            if not success:
                print(f"❌ Failed to send verse to STTM: {vid}")
                
            last_committed_id = vid
            last_commit_idx = idx
            persistence_counter = 0
            log("committed", chunk_ms=chunk_ms, asr_text=asr_text,
                best_verse_id=vid, best_score='auto', committed=True)
            print(f"=== MATCHED ({strong_type.upper()}): id={vid}, orig_text='{candidate['orig_text']}' ===")
            transcript_buffer.clear()
    else:
        persistence_counter = 0

def is_generic_line(idx):
    line = corpus_orig[idx]
    toks = set(line.replace("॥","").split())
    return len(toks & GENERIC_TOKENS) > 0 or len(line.strip()) <= 6

def transcribe_chunk(audio_chunk):
    global audio_buffer
    chunk_input = np.concatenate([audio_buffer, audio_chunk])
    audio_buffer = audio_chunk[-int(SAMPLE_RATE*OVERLAP):]
    inputs = processor(chunk_input, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    txt = processor.batch_decode(pred_ids)[0]
    txt = txt.replace("ੱ","").strip()
    for w in txt.split():
        transcript_buffer.append(w)
    return txt, (time.time()-t0)*1000.0

def audio_callback(indata, frames, time_info, status):
    global last_commit_idx, anchor_queue, anchor_mode

    start_all = time.perf_counter()
    if status:
        print("Audio status:", status)
    t_asr0 = time.perf_counter()
    audio_chunk = indata[:,0].astype(np.float32)
    asr_text, ms = transcribe_chunk(audio_chunk)
    t_asr1 = time.perf_counter()

    logger.info(f"[TIMER] ASR: {(t_asr1-t_asr0)*1000:.1f} ms")
    logger.info(f"Transcribed: {asr_text}")

    N = 10
    asr_tokens = list(transcript_buffer)[-N:]

    MIN_ANCHOR_WORDS = 10
    if MODE == "paath" and anchor_mode:
        if len(transcript_buffer) < MIN_ANCHOR_WORDS:
            print("[ANCHOR] Waiting for buffer to fill before anchoring...")
            return
        fuzzy = select_best_fuzzy_anywhere(transcript_buffer)
        anchor_queue.append(fuzzy[1])
        anchor_counter = Counter(anchor_queue)
        most_common_idx, count = anchor_counter.most_common(1)[0]
        if not is_generic_line(most_common_idx) and most_common_idx is not None and most_common_idx >= 0:
            if fuzzy[0] >= ANCHOR_STRONG_SCORE or (count >= ANCHOR_MIN_FAST_AGREEMENT and fuzzy[0] >= ANCHOR_WEAK_SCORE):
                print(f"✅ [ANCHOR FINAL] idx={most_common_idx}, orig='{corpus_orig[most_common_idx]}', agreement={count}, score={fuzzy[0]:.2f}")
                last_commit_idx = most_common_idx
                anchor_mode = False
                anchor_queue.clear()
                
                # Check if STTM sync is already initialized by orchestrator
                if not (sttm_sync and sttm_sync.connected):
                    # Initialize STTM sync connection after anchor is set
                    success = initialize_sttm_sync()
                    if success:
                        print("✅ STTM sync connection established successfully")
                    else:
                        print("❌ Failed to establish STTM sync connection")
                else:
                    print("✅ Using pre-established STTM sync connection")
                
                return
            else:
                print(f"[ANCHOR WAIT] idx={most_common_idx}, orig='{corpus_orig[most_common_idx]}', agreement={count}, score={fuzzy[0]:.2f}")
                return

    expected_next = last_commit_idx + 1
    lookahead_words = LEADING_TRIGGER_WORDS
    if expected_next < len(verse_ids):
        buf_list = list(transcript_buffer)
        if len(buf_list) >= lookahead_words:
            buffer_start = " ".join(buf_list[-lookahead_words:])
            next_line = corpus_asr[expected_next]
            score = ensemble_score(buffer_start, next_line[:len(buffer_start)])
            if score > LEADING_TRIGGER_SCORE:
                print(f"[LEADING TRIGGER] idx={expected_next}, line='{corpus_orig[expected_next]}'")
                last_commit_idx = expected_next
                
                # Send verse to STTM using the sync controller
                vid = verse_ids[expected_next]
                success = send_verse_to_sttm(vid)
                if success:
                    print(f"✅ Sent verse to STTM (leading trigger): {vid}")
                else:
                    print(f"❌ Failed to send verse to STTM (leading trigger): {vid}")
                    
                transcript_buffer.clear()
                return

    allowed_indices = get_paath_allowed_indices(last_commit_idx, len(verse_ids))
    fuzzy = select_best_fuzzy_paath(transcript_buffer, allowed_indices)
    lcs = lcs_best(asr_tokens, corpus_tokens, corpus_orig, allowed_indices)
    seqm = seqmatcher_best(asr_tokens, corpus_asr, corpus_orig, allowed_indices)
    lev = levenshtein_best(asr_tokens, corpus_asr, allowed_indices)
    print(f"Fuzzy: idx={fuzzy[1]}, score={fuzzy[0]:.2f}, line='{fuzzy[2]['orig_text'] if fuzzy[2] else ''}'")
    print(f"LCS: idx={lcs[1]}, lcs={lcs[0]}, line='{lcs[2]}'")
    print(f"SequenceMatcher: idx={seqm[1]}, ratio={seqm[0]:.3f}, line='{seqm[2]}'")
    print(f"Levenshtein: idx={lev[1]}, score={lev[0]:.3f}, line='{lev[2]}'")
    winner_type, winner_idx = pick_strongest_paath(fuzzy, lcs, seqm, lev, asr_tokens)
    print(f"=> CHOSEN: {winner_type.upper()} idx={winner_idx} orig='{corpus_orig[winner_idx] if winner_idx != -1 else ''}'")
    if winner_idx not in allowed_indices or abs(winner_idx - (last_commit_idx + 1)) > 2:
        print("[DRIFT] Out of sequential window, anchor resumes!")
        anchor_mode = True
        anchor_queue.clear()
        return
    maybe_commit(winner_type, winner_idx, ms, asr_text)
    print(f"[TIMER] Chunk processed in {time.perf_counter()-start_all:.1f} ms")

def main():
    """Main entry point for the agent"""
    chunk = int(SAMPLE_RATE*CHUNK_DURATION)
    stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=chunk, callback=audio_callback)
    stream.start()
    print("MMS local agent running...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean up resources
        stream.stop()
        stream.close()
        
        # Disconnect from STTM if connected
        if sttm_sync and sttm_sync.connected:
            print("Disconnecting from STTM...")
            sttm_sync.disconnect()

if __name__ == "__main__":
    main()
