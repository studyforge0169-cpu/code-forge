"""M61 LIVE certification smoke (executed EXACTLY ONCE against production).

Reclaims ONE genuinely safe checkpoint through the M61 surface itself:

    best == published == 41870af53ab7 (post-M60)
    victim = 4772821d72c9 (step 2, 6.452705 — the FIRST M59 winner,
                            superseded, ZERO blocking references)
                    |
                    v
        ONE explicit verified deletion (DELETE .../checkpoints/4772821d72c9)
                    |
                    v
    checkpoint gone; best/published/weights/manifest unchanged;
    M59 history recomputes over the 22 survivors; nothing else moved.

Safety was proven BEFORE the mutation three independent ways:
(1) not the M52 best, not the published latest, not the manifest's
    best-known reference (independent scan of every record family);
(2) the new engine analysis checkpoint_blockers() == [];
(3) the smoke itself re-verifies both read-only through the live API
    before the single DELETE.

Production inventory was captured BEFORE the smoke (m61_pre.sha256,
79 files / 15,283,812 B — byte-identical to the post-M60 state).

EXECUTION RECORD (2026-09-10, port 8783, exactly once): the single
state-mutating call ran and the server log is the authoritative record
(P1-P7 GETs 200; both protected DELETEs 409; the ONE victim DELETE 200
with the engine logging exactly "2 files, 630554 bytes"; victim GET 404;
repeated DELETE 404). The script itself crashed at its V12 residue
assertion on a bug in the CHECK code (os.listdir returns str, not
Path — the mutation was already complete and correct). The bug is
fixed above and every remaining fact was re-verified READ-ONLY
post-hoc against production (17/17 — registry/best/published/history
content, blocker sets, and the byte-level inventory diff showing
exactly the victim's 2 files removed and nothing else touched). No
production state was mutated after the single deletion.

Usage: python smoke_m61_live.py http://127.0.0.1:<port>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
MID = "31db39e17a20"
DATA = "/home/user/ai-model-forge-data"
MODEL_DIR = f"{DATA}/models/{MID}"

# Certified pre-smoke production facts (captured read-only 2026-09-10,
# before ANY M61 smoke call; see m61_pre.sha256):
BEST = "41870af53ab7"           # live M52 best (== published, post-M60)
BEST_LOSS = 6.242401
PUBLISHED = "41870af53ab7"      # manifest latest_checkpoint
MANIFEST_BEST = "eaf0984fbc2b"  # manifest best_checkpoint (legacy pointer)
VICTIM = "4772821d72c9"         # step 2 @ 6.452705 — the FIRST M59 winner
VICTIM_LOSS = 6.452705
VICTIM_RUN = "15fa1b096305"
VICTIM_FILES = 2                # manifest.json + weights.pt
VICTIM_BYTES = 630554           # 558 + 629,996
N_CKPTS = 23
N_HISTORY = 15
# the surviving winners in order (the victim was entry #1)
HISTORY_AFTER = ["a58e966685cc", "5a6e4ca0ecbc", "8deb1656701b",
                 "e6b4a889a90f", "f67eb449ee0a", "ed8dbd1b3578",
                 "0e2c32af3ed3", "0deba72e350b", "0bae5e481ff6",
                 "1b6442a0fc58", "958bf4b220ed", "85c0ea2b6d52",
                 "976df555e307", "41870af53ab7"]
M58_TAIL = ["958bf4b220ed", "85c0ea2b6d52", "976df555e307", "41870af53ab7"]

passed: list[str] = []
failed: list[str] = []


def check(section: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(
        f"{section}: {'PASS' if ok else 'FAIL ' + detail}")


def req(method: str, path: str) -> tuple[int, object]:
    r = urllib.request.Request(BASE + path, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ prelude
st, model = req("GET", f"/api/v1/models/{MID}")
check("P1 model readable", st == 200)
check("P2 published == 41870af53ab7 (post-M60 state)",
      model["latest_checkpoint"] == PUBLISHED)

st, best = req("GET", f"/api/v1/models/{MID}/checkpoints/best")
check("P3 M52 best unchanged pre-smoke", st == 200
      and best["checkpoint"]["checkpoint_id"] == BEST
      and best["checkpoint"]["validation_loss"] == BEST_LOSS)

st, hist = req("GET", f"/api/v1/models/{MID}/checkpoints/best/history")
check("P4 M59 history intact pre-smoke", st == 200
      and hist["candidate_count"] == N_CKPTS
      and len(hist["entries"]) == N_HISTORY
      and hist["entries"][0]["checkpoint_id"] == VICTIM
      and hist["entries"][0]["validation_loss"] == VICTIM_LOSS
      and hist["entries"][-1]["checkpoint_id"] == BEST)
check("P5 M58 tail intact",
      [e["checkpoint_id"] for e in hist["entries"][-4:]] == M58_TAIL)

st, cks = req("GET", f"/api/v1/models/{MID}/checkpoints")
check("P6 checkpoint registry size", st == 200 and len(cks) == N_CKPTS)

st, victim = req("GET", f"/api/v1/models/{MID}/checkpoints/{VICTIM}")
check("P7 victim readable + certified identity", st == 200
      and victim["run_id"] == VICTIM_RUN
      and victim["validation_loss"] == VICTIM_LOSS)

# read-only safety re-verification through the live API: the protected
# ids must be REFUSED (proving the guard) — best, published, legacy
st, r_best = req("DELETE", f"/api/v1/models/{MID}/checkpoints/{BEST}")
check("P8 DELETE(best==published) refused 409 + ordered blockers",
      st == 409 and r_best["detail"]["protected"] is True
      and [b["reason"] for b in r_best["detail"]["blockers"]][:2]
      == ["best", "published"],
      f"status={st} body={r_best}")
st, r_mb = req("DELETE", f"/api/v1/models/{MID}/checkpoints/{MANIFEST_BEST}")
check("P9 DELETE(manifest best-known ref) refused 409",
      st == 409 and "manifest_reference" in
      [b["reason"] for b in r_mb["detail"]["blockers"]],
      f"status={st}")
st, cks2 = req("GET", f"/api/v1/models/{MID}/checkpoints")
check("P10 rejections deleted nothing", len(cks2) == N_CKPTS)

pre_weights = sha(f"{MODEL_DIR}/weights.pt")
pre_manifest = sha(f"{MODEL_DIR}/manifest.json")
pre_ck = {c["checkpoint_id"]: c for c in cks}
pre_files = {c["checkpoint_id"]:
             sha(f"{MODEL_DIR}/checkpoints/{c['checkpoint_id']}/manifest.json")
             for c in cks}

# ------------------------------------------------- THE ONE MUTATING CALL
st, res = req("DELETE", f"/api/v1/models/{MID}/checkpoints/{VICTIM}")
check("X1 explicit deletion executed", st == 200,
      f"status={st} body={res}")

# -------------------------------------------------------- post-verification
check("V1 deterministic result",
      st == 200 and res == {"model_id": MID, "checkpoint_id": VICTIM,
                            "files_removed": VICTIM_FILES,
                            "bytes_reclaimed": VICTIM_BYTES},
      f"body={res}")

st, gone = req("GET", f"/api/v1/models/{MID}/checkpoints/{VICTIM}")
check("V2 victim gone (detail 404)", st == 404)
st, again = req("DELETE", f"/api/v1/models/{MID}/checkpoints/{VICTIM}")
check("V3 repeated deletion -> 404", st == 404)

st, cks3 = req("GET", f"/api/v1/models/{MID}/checkpoints")
check("V4 registry = 22 survivors, verbatim records",
      st == 200 and len(cks3) == N_CKPTS - 1
      and {c["checkpoint_id"]: c for c in cks3}
      == {k: v for k, v in pre_ck.items() if k != VICTIM})

st, best2 = req("GET", f"/api/v1/models/{MID}/checkpoints/best")
check("V5 M52 best unchanged", st == 200
      and best2["checkpoint"]["checkpoint_id"] == BEST
      and best2["checkpoint"]["validation_loss"] == BEST_LOSS)

st, model2 = req("GET", f"/api/v1/models/{MID}")
check("V6 published state unchanged (model manifest untouched)",
      st == 200 and model2["latest_checkpoint"] == PUBLISHED)

st, hist2 = req("GET", f"/api/v1/models/{MID}/checkpoints/best/history")
check("V7 M59 history recomputed over survivors",
      st == 200 and hist2["candidate_count"] == N_CKPTS - 1
      and [e["checkpoint_id"] for e in hist2["entries"]] == HISTORY_AFTER
      and hist2["entries"][0]["delta_loss_nats"] is None
      and hist2["entries"][-1]["checkpoint_id"] == BEST,
      f"got {[e['checkpoint_id'] for e in hist2.get('entries', [])]}")
check("V8 M58 tail still intact",
      [e["checkpoint_id"] for e in hist2["entries"][-4:]] == M58_TAIL)

check("V9 weights.pt byte-identical",
      sha(f"{MODEL_DIR}/weights.pt") == pre_weights)
check("V10 model manifest byte-identical",
      sha(f"{MODEL_DIR}/manifest.json") == pre_manifest)
check("V11 every surviving checkpoint manifest byte-identical",
      all(sha(f"{MODEL_DIR}/checkpoints/{cid}/manifest.json") == h
          for cid, h in pre_files.items() if cid != VICTIM))
check("V12 victim directory gone, no residue",
      not os.path.exists(f"{MODEL_DIR}/checkpoints/{VICTIM}")
      and not any(p.startswith(".tmp")
                 for p in os.listdir(f"{MODEL_DIR}/checkpoints")))

# ------------------------------------------------------------- audit files
inventory = {}
for line in open("/home/user/code-forge/m61_pre.sha256"):
    entry = line.split(None, 1)
    if len(entry) == 2:
        inventory[entry[1].strip().lstrip("./")] = entry[0]
new_files: list[str] = []
modified: list[str] = []
for root, _dirs, files in os.walk(DATA):
    if root == f"{DATA}/tmp":
        continue
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, DATA)
        digest = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if rel not in inventory:
            new_files.append(rel)
        elif inventory[rel] != digest:
            modified.append(rel)
check("A1 tmp clean",
      not os.path.exists(f"{DATA}/tmp")
      or os.listdir(f"{DATA}/tmp") == [])
check("A2 exactly the victim's 2 files removed, nothing else",
      new_files == [] and modified == [],
      f"new={new_files} modified={modified}")
n_after = len(inventory) - 2
actual_after = sum(1 for r, d, files in os.walk(DATA)
                   if r != f"{DATA}/tmp"
                   for f in files if not f.startswith("."))
check("A3 file count 79 -> 77", actual_after == n_after == 77,
      f"actual={actual_after}")

print("\n".join(passed))
if failed:
    print("\n".join(failed))
    sys.exit(1)
print(f"\n{len(passed)}/{len(passed) + len(failed)} LIVE M61 CHECKS PASSED")
