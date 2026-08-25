"""Batched selective Hypersim downloader (adapted from ml-hypersim contrib/99991, MIT).
Reads download_manifest.json {scene: [member-suffixes]}, opens each scene zip remotely
(HTTP range reads), extracts exactly the needed members. Parallel over scenes, resumable.
"""
import os, sys, json, zipfile, threading, queue, time
import requests

zipfile.ZipExtFile.MIN_READ_SIZE = 2 ** 20
BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"
OUT  = "data/hypersim_omni3d/raw"
N_WORKERS = 12

class WebFile:
    def __init__(self, url, session):
        r = session.head(url); r.raise_for_status()
        self.size = int(r.headers["content-length"])
        self.url, self.session, self.offset = url, session, 0
    def seekable(self): return True
    def tell(self): return self.offset
    def seek(self, offset, whence=0):
        if whence == 0: self.offset = offset
        elif whence == 1: self.offset = min(self.offset + offset, self.size)
        elif whence == 2: self.offset = max(0, self.size + offset)
    def read(self, n=None):
        n = self.size - self.offset if n is None else min(n, self.size - self.offset)
        if n <= 0: return b""
        h = {"Range": f"bytes={self.offset}-{self.offset+n-1}"}
        r = self.session.get(self.url, headers=h); r.raise_for_status()
        data = r.content; self.offset += len(data); return data

manifest = json.load(open("data/hypersim_omni3d/download_manifest.json"))
scenes = sorted(manifest)
q = queue.Queue()
for s in scenes: q.put(s)
lock = threading.Lock()
stats = {"scenes_done": 0, "files": 0, "missing": [], "errors": []}

def worker():
    session = requests.session()
    while True:
        try: scene = q.get_nowait()
        except queue.Empty: return
        wanted = manifest[scene]
        # resume: skip scene if all files present
        if all(os.path.isfile(os.path.join(OUT, scene, w)) and
               os.path.getsize(os.path.join(OUT, scene, w)) > 0 for w in wanted):
            with lock: stats["scenes_done"] += 1
            q.task_done(); continue
        for attempt in range(3):
            try:
                z = zipfile.ZipFile(WebFile(BASE.format(scene), session))
                members = {}
                for info in z.infolist():
                    for w in wanted:
                        if info.filename.endswith(w): members[w] = info
                for w in wanted:
                    dst = os.path.join(OUT, scene, w)
                    if os.path.isfile(dst) and os.path.getsize(dst) > 0: continue
                    if w not in members:
                        with lock: stats["missing"].append(f"{scene}/{w}")
                        continue
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with z.open(members[w]) as src, open(dst + ".tmp", "wb") as f:
                        f.write(src.read())
                    os.replace(dst + ".tmp", dst)
                    with lock: stats["files"] += 1
                break
            except Exception as e:
                if attempt == 2:
                    with lock: stats["errors"].append(f"{scene}: {e}")
                time.sleep(2 * (attempt + 1))
        with lock:
            stats["scenes_done"] += 1
            if stats["scenes_done"] % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] scenes {stats['scenes_done']}/{len(scenes)} files {stats['files']} missing {len(stats['missing'])} errors {len(stats['errors'])}", flush=True)
        q.task_done()

threads = [threading.Thread(target=worker, daemon=True) for _ in range(N_WORKERS)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"DONE scenes={stats['scenes_done']} files={stats['files']} missing={len(stats['missing'])} errors={len(stats['errors'])}")
for m in stats["missing"][:20]: print("  missing:", m)
for e in stats["errors"][:20]: print("  error:", e)
json.dump({"missing": stats["missing"], "errors": stats["errors"]},
          open("data/hypersim_omni3d/fetch_report.json", "w"))
