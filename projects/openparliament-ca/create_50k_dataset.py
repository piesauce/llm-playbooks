# Creates a dataset of 50k samples to train models. Data is exported to eval_sampler_balanced_50k_preview10_stop_at_targets.py
import re, time, csv, json, random
from html import unescape
from typing import Dict, Any, Iterable, List, Optional
import boto3
from botocore.config import Config
import ollama

# ========= CONFIG =========
BUCKET = "canada-training"
PREFIX = "politician-speeches/"

MODEL = "qwen3:14b"
MAX_RETRIES = 3
MAX_PREDICT = 64

CHARS_PER_TOKEN = 4
MARGIN_TOKENS = 200
CTX_CAP = 1024

# ----- Balanced target totals -----
TOTAL_TARGET = 50_000
CLASSES = ("political promise", "goal/expectation", "neither")

BASE = TOTAL_TARGET // 3          # 16666
REMA = TOTAL_TARGET % 3           # 2
PER_CLASS_TARGETS = {
    CLASSES[i]: BASE + (1 if i < REMA else 0) for i in range(len(CLASSES))
}   # {'political promise': 16667, 'goal/expectation': 16667, 'neither': 16666}

OUTPUT_CSV = "balanced_50k_preview10.csv"
RNG_SEED = 42
LOG_EVERY = 500  # progress heartbeat

# ========= CLIENTS =========
s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
client = ollama.Client()
random.seed(RNG_SEED)

PROMPT_TMPL = """
You are a political scientist analyzing speeches of members of parliament. You are trying to classify each paragraph of text into one of three categories: 
- "political promise"
- "goal/expectation"
- "neither"

Definitions:
- A political promise contains both (a) definitive language ("we will," "we pledge") AND (b) an explicit *actor* committing to action (e.g., "our government will introduce a law"). The actor must be related to a person, or a group of people.
- A goal/expectation uses predictive or aspirational language ("will happen," "is expected to") but does NOT explicitly commit the speaker or their party to act.
- Neither applies when the text does not clearly indicate a promise, goal, or expectation. Text with clarifying questions should be considered neither.

Examples:

Text: "We will introduce legislation to strengthen privacy rights this fall."
Classification: political promise

Text: "Our government commits to building 10,000 affordable housing units by 2027."
Classification: political promise

Text: "The unemployment rate will fall to 5% by next year."
Classification: goal/expectation

Text: "Our hope is that Canada becomes a global leader in artificial intelligence."
Classification: goal/expectation

Text: "Yesterday, the finance committee met to review budget allocations."
Classification: neither

Text: "Mr. Speaker, I thank my colleague for his thoughtful question."
Classification: neither

Think step by step internally, but only return one line in this exact format:
Classification: <political promise | goal/expectation | neither> because <short reason>.

Metadata:
politician_id: {politician_id}
politician_name: {politician_name}
document_id: {document_id}
time: {time}
h1_en: {h1_en}
h2_en: {h2_en}

SPEECH:
\"\"\"{content_en}\"\"\"
""".strip()

# ----- helpers -----
def now(): return time.time()
def tdiff(t0): return f"{(time.time()-t0):.2f}s"

def estimate_tokens(text: str, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    return max(1, len(text) // chars_per_token)

def safe_num_ctx(prompt_text: str, speech_text: str, margin_tokens: int = MARGIN_TOKENS, cap: int = CTX_CAP) -> int:
    total = estimate_tokens(prompt_text) + estimate_tokens(speech_text) + margin_tokens
    return min(total, cap)

def iter_keys(bucket: str, prefix: str) -> Iterable[str]:
    """Yield ALL JSON keys under the prefix (across every S3 pagination page)."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if k.endswith("/") or not k.lower().endswith(".json"):
                continue
            yield k

def get_json(bucket: str, key: str) -> Dict[str, Any]:
    res = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(res["Body"].read().decode("utf-8", errors="replace"))

def strip_html(html_text: str) -> str:
    text = unescape(html_text or "")
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()

def first_n_words(text: str, n: int = 10) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[:n])

def extract_label(model_output: str) -> str:
    m = re.search(r"Classification:\s*([^\n]+)", model_output, flags=re.I)
    if m:
        raw = m.group(1).lower()
        if "political" in raw and "promise" in raw: return "political promise"
        if "goal" in raw or "expectation" in raw:  return "goal/expectation"
        if "neither" in raw:                        return "neither"
    for lbl in CLASSES:
        if lbl in (model_output or "").lower():
            return lbl
    return "neither"

def classify_one(formatted_prompt: str, ctx: int) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream = client.generate(
                model=MODEL,
                prompt=formatted_prompt,
                options={"num_ctx": ctx, "num_predict": MAX_PREDICT},
                stream=True,
            )
            out = []
            for ch in stream:
                if ch.response:
                    out.append(ch.response)
            return extract_label("".join(out))
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[ERR] model failed after {MAX_RETRIES} tries: {e}", flush=True)
                return "neither"
            backoff = 2 ** (attempt - 1)
            print(f"[WARN] model error ({e}); retrying in {backoff}s...", flush=True)
            time.sleep(backoff)

# ----- balanced reservoir logic -----
class BalancedReservoir:
    """
    Keep target[c] items per class with reservoir sampling.
    Once all targets are met, you can stop the crawl.
    """
    def __init__(self, cls_targets: Dict[str, int]):
        self.targets = dict(cls_targets)
        self.buckets: Dict[str, List[dict]] = {c: [] for c in self.targets}
        self.seen_counts: Dict[str, int] = {c: 0 for c in self.targets}

    def maybe_add(self, cls: str, row: dict):
        if cls not in self.targets:
            return
        self.seen_counts[cls] += 1
        seen = self.seen_counts[cls]
        k = self.targets[cls]
        bucket = self.buckets[cls]
        if len(bucket) < k:
            bucket.append(row)
        else:
            # with probability k/seen, replace a random existing element
            j = random.randint(1, seen)
            if j <= k:
                bucket[j-1] = row

    def is_full(self) -> bool:
        return all(len(self.buckets[c]) >= self.targets[c] for c in self.targets)

    def total(self) -> int:
        return sum(len(self.buckets[c]) for c in self.targets)

# ----- main -----
def main():
    targets = PER_CLASS_TARGETS
    res = BalancedReservoir(targets)
    files_seen = 0

    t_start = now()
    for key in iter_keys(BUCKET, PREFIX):
        if res.is_full():
            break
        files_seen += 1

        try:
            blob = get_json(BUCKET, key)
        except Exception as e:
            print(f"[SKIP] {key}: JSON load error: {e}", flush=True)
            continue

        pol_id_raw = blob.get("politician_id", "")
        try:
            pol_id = int(float(pol_id_raw))
        except Exception:
            pol_id = pol_id_raw
        pol_name = blob.get("name", "")

        speeches = (blob.get("speeches") or [])
        for s in speeches:
            if res.is_full():
                break

            content = strip_html(s.get("content_en") or "")
            if not content:
                continue

            formatted_prompt = PROMPT_TMPL.format(
                politician_id=pol_id,
                politician_name=pol_name,
                document_id=s.get("document_id", ""),
                time=s.get("time", "") or "",
                h1_en=s.get("h1_en", "") or "",
                h2_en=s.get("h2_en", "") or "",
                content_en=content
            )

            ctx = safe_num_ctx(PROMPT_TMPL, content)
            label = classify_one(formatted_prompt, ctx)

            row = {
                "s3_key": key,
                "document_id": s.get("document_id", ""),
                "politician_id": pol_id,
                "politician_name": pol_name,
                "time": s.get("time", "") or "",
                "h1_en": s.get("h1_en", "") or "",
                "h2_en": s.get("h2_en", "") or "",
                "content_preview10": first_n_words(content, 10),
                "classification": label
            }

            if label in CLASSES:
                res.maybe_add(label, row)

                if res.total() % LOG_EVERY == 0 and res.total() > 0:
                    status = " | ".join(
                        f"{c}:{len(res.buckets[c])}/{targets[c]}" for c in CLASSES
                    )
                    print(f"[PROGRESS] kept={res.total()}/{TOTAL_TARGET}  [{status}] | files_seen={files_seen}", flush=True)

    elapsed = tdiff(t_start)

    # Write out the balanced sample (final buckets)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "s3_key","document_id","politician_id","politician_name",
            "time","h1_en","h2_en","content_preview10","classification"
        ])
        for c in CLASSES:
            for r in res.buckets[c]:
                w.writerow([
                    r["s3_key"], r["document_id"], r["politician_id"], r["politician_name"],
                    r["time"], r["h1_en"], r["h2_en"], r["content_preview10"], r["classification"]
                ])

    print("\n===== DONE (stopped at targets) =====", flush=True)
    print(f"Files scanned:   {files_seen}", flush=True)
    for c in CLASSES:
        print(f"  {c}: {len(res.buckets[c])}/{targets[c]}", flush=True)
    print(f"Total kept:      {res.total()} (target {TOTAL_TARGET})", flush=True)
    print(f"Elapsed:         {elapsed}", flush=True)

if __name__ == "__main__":
    main()
