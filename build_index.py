
import os, json
from collections import defaultdict
from datasets import load_dataset

OUT_DIR = "local_banidb"
HF_TOKEN = os.environ.get("HF_TOKEN") 

def build_index():
    ds = load_dataset("jssaluja/verse_dataset", token=HF_TOKEN, split="train")

    line_store = {}
    inverted = defaultdict(set)

    for row in ds:
        vid = row["verse_id"]
        asr_text = row["asr_text"]
        orig_text = row["orig_text"]
        line_store[vid] = {
            "asr_text": asr_text,
            "orig_text": orig_text,
            "page": row.get("page"),
            "line": row.get("line")
        }
        for tok in [t for t in asr_text.split() if t.strip()]:
            inverted[tok].add(vid)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR,"line_store.json"),"w",encoding="utf8") as f:
        json.dump(line_store,f,ensure_ascii=False)
    with open(os.path.join(OUT_DIR,"inverted.json"),"w",encoding="utf8") as f:
        json.dump({k:list(v) for k,v in inverted.items()},f,ensure_ascii=False)
    print("Index built:", len(line_store), "verses,", len(inverted), "tokens")

if __name__ == "__main__":
    build_index()
