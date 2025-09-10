
import csv, os
from datetime import datetime
import pandas as pd
from threading import Lock
from flask import Flask

OUT = "telemetry.csv"
lock = Lock()
FIELDS = ["ts","event","chunk_ms","asr_text","best_verse_id","best_score","committed"]

def init():
    if not os.path.exists(OUT):
        with open(OUT,"w",newline="",encoding="utf8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

def log(event, chunk_ms=None, asr_text="", best_verse_id=None, best_score=None, committed=False):
    init()
    row = {"ts": datetime.utcnow().isoformat(), "event": event, "chunk_ms": chunk_ms,
           "asr_text": asr_text, "best_verse_id": best_verse_id, "best_score": best_score,
           "committed": committed}
    with lock:
        with open(OUT,"a",newline="",encoding="utf8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

app = Flask(__name__)
@app.route("/telemetry/summary")
def summary():
    if not os.path.exists(OUT):
        return {"rows":0}
    df = pd.read_csv(OUT)
    return {"rows": len(df), "avg_chunk_ms": float(df["chunk_ms"].dropna().mean() or 0),
            "committed_pct": float((df["committed"].sum() / max(1,len(df))) * 100)}

@app.route("/telemetry/recent")
def recent():
    if not os.path.exists(OUT):
        return []
    return pd.read_csv(OUT).tail(200).to_dict(orient="records")

if __name__ == "__main__":
    app.run(port=9001)
