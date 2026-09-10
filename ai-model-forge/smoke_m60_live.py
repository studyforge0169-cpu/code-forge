"""M60 LIVE certification smoke (executed EXACTLY ONCE against production).

Closes the production drift with the M60 surface itself:

    best (41870af53ab7 @ 6.242401) != latest/published (eaf0984fbc2b)
                    |
                    v
        ONE inline workflow: PUBLISH(best)
                    |
                    v
    published/latest == best  (the M60 invariant)

Read-only prelude + EXACTLY ONE state-mutating call (the workflow run) +
read-only post-verification against the LIVE service and the filesystem.
Production inventory was captured BEFORE the smoke (m60_pre.sha256,
78 files, byte-identical to m59_pre.sha256 — M59 added nothing).

Usage: python smoke_m60_live.py http://127.0.0.1:<port>
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
MID = "31db39e17a20"
DATA = "/home/user/ai-model-forge-data"
MODEL_DIR = f"{DATA}/models/{MID}"

# Certified pre-smoke production facts (captured read-only 2026-09-10,
# before ANY M60 smoke call; see m60_pre.sha256):
BEST = "41870af53ab7"          # live M52 argmin
BEST_LOSS = 6.242401           # persisted validation_loss of BEST
LATEST = "eaf0984fbc2b"        # published pointer BEFORE the smoke
N_CKPTS = 23
N_HISTORY = 15                 # M59 movements
M58_TAIL = 4                   # B7: the last FOUR movements are M58's

passed: list[str] = []
failed: list[str] = []


def check(section: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(
        f"{section}: {'PASS' if ok else 'FAIL ' + detail}")


def req(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ prelude
st, model = req("GET", f"/api/v1/models/{MID}")
check("P1 model readable", st == 200)
check("P2 drift EXISTS pre-smoke",
      model["latest_checkpoint"] == LATEST and LATEST != BEST,
      f"latest={model['latest_checkpoint']}")

st, best = req("GET", f"/api/v1/models/{MID}/checkpoints/best")
check("P3 M52 best is BEST", st == 200
      and best["checkpoint"]["checkpoint_id"] == BEST
      and best["checkpoint"]["validation_loss"] == BEST_LOSS)

st, hist = req("GET", f"/api/v1/models/{MID}/checkpoints/best/history")
check("P4 M59 history intact pre-smoke", st == 200
      and hist["candidate_count"] == N_CKPTS
      and len(hist["entries"]) == N_HISTORY
      and hist["entries"][-1]["checkpoint_id"] == BEST)
check("P5 M58 tail intact",
      [e["checkpoint_id"] for e in hist["entries"][-M58_TAIL:]] ==
      ["958bf4b220ed", "85c0ea2b6d52", "976df555e307", "41870af53ab7"],
      "tail mismatch")

st, cks = req("GET", f"/api/v1/models/{MID}/checkpoints")
check("P6 checkpoint registry size", st == 200 and len(cks) == N_CKPTS)

pre_best_w = sha(f"{MODEL_DIR}/checkpoints/{BEST}/weights.pt")
pre_best_m = sha(f"{MODEL_DIR}/checkpoints/{BEST}/manifest.json")
pre_latest_w = sha(f"{MODEL_DIR}/checkpoints/{LATEST}/weights.pt")
pre_latest_m = sha(f"{MODEL_DIR}/checkpoints/{LATEST}/manifest.json")
pre_best_sha_field = json.loads(
    open(f"{MODEL_DIR}/checkpoints/{BEST}/manifest.json").read())["weights_sha256"]
n_wf_pre = len([d for d in __import__("os").listdir(f"{MODEL_DIR}/workflows")])
check("P7 source hashes captured", bool(pre_best_sha_field))

# ------------------------------------------------- THE ONE MUTATING CALL
st, rec = req("POST", "/api/v1/workflows/run", {
    "name": "m60-live-publish-best",
    "model_id": MID,
    "description": "M60 live certification: PUBLISH(best) closing the "
                   "best!=latest drift through the M3 rollback machinery",
    "stages": [{"stage_id": "publish_best", "type": "publish",
                "publish": {"publish_from_best": True}}]})
check("X1 PUBLISH(best) executed", st == 200, f"status={st} body={rec}")

# -------------------------------------------------------- post-verification
pub = rec["plan"]["stages"][0]["publish"] if st == 200 else {}
art = rec["stages"][0]["artifact"] if st == 200 else {}
check("V1 workflow completed", st == 200 and rec["status"] == "completed")
check("V2 plan pinned BEST before execution",
      pub.get("publish_from_best") is True
      and pub.get("resolved_checkpoint_id") == BEST
      and pub.get("checkpoint_id") is None)
check("V3 artifact identifies BEST (reference-only)",
      art.get("kind") == "publication"
      and art.get("artifact_id") == BEST
      and art.get("checkpoint_id") == BEST
      and art.get("state_hash") == pre_best_sha_field
      and art.get("final_validation_loss") == BEST_LOSS
      and art.get("result_hash") is None)

st, model2 = req("GET", f"/api/v1/models/{MID}")
check("V4 THE INVARIANT: published == best",
      st == 200 and model2["latest_checkpoint"] == BEST,
      f"latest={model2.get('latest_checkpoint')}")

st, best2 = req("GET", f"/api/v1/models/{MID}/checkpoints/best")
check("V5 M52 best endpoint unchanged",
      st == 200 and best2["checkpoint"]["checkpoint_id"] == BEST
      and best2["checkpoint"]["validation_loss"] == BEST_LOSS)

st, hist2 = req("GET", f"/api/v1/models/{MID}/checkpoints/best/history")
check("V6 M59 history unchanged",
      st == 200 and hist2 == hist,
      "history changed after publication")

st, cks2 = req("GET", f"/api/v1/models/{MID}/checkpoints")
check("V7 no new checkpoint created",
      st == 200 and len(cks2) == N_CKPTS)

check("V8 source checkpoint BEST immutable",
      sha(f"{MODEL_DIR}/checkpoints/{BEST}/weights.pt") == pre_best_w
      and sha(f"{MODEL_DIR}/checkpoints/{BEST}/manifest.json") == pre_best_m)
check("V9 former latest LATEST immutable",
      sha(f"{MODEL_DIR}/checkpoints/{LATEST}/weights.pt") == pre_latest_w
      and sha(f"{MODEL_DIR}/checkpoints/{LATEST}/manifest.json") == pre_latest_m)

# published weights == BEST's verified state (read-only file comparison)
import torch  # noqa: E402
state = torch.load(f"{MODEL_DIR}/checkpoints/{BEST}/weights.pt",
                   map_location="cpu", weights_only=True)
cur = torch.load(f"{MODEL_DIR}/weights.pt", map_location="cpu",
                 weights_only=True)
check("V10 published weights are BEST's state",
      set(state) == set(cur) and all(torch.equal(state[k], cur[k])
                                     for k in state))
side = json.loads(open(f"{MODEL_DIR}/weights.sha256").read())["sha256"]
check("V11 weights sidecar hash matches file",
      side == sha(f"{MODEL_DIR}/weights.pt"))

n_wf_post = len([d for d in __import__("os").listdir(f"{MODEL_DIR}/workflows")])
check("V12 exactly ONE new workflow record",
      n_wf_post == n_wf_pre + 1,
      f"{n_wf_pre} -> {n_wf_post}")

st, wfs = req("GET", f"/api/v1/models/{MID}/workflows")
m60 = [w for w in wfs if w["workflow_id"] == rec.get("workflow_id")]
check("V13 record listed with pinned lineage",
      st == 200 and len(m60) == 1
      and m60[0]["plan"]["stages"][0]["publish"]["resolved_checkpoint_id"]
      == BEST
      and m60[0]["status"] == "completed")

# ------------------------------------------------------------- audit files
import os  # noqa: E402
inventory = {}
for line in open("/home/user/code-forge/m60_pre.sha256"):
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
check("A2 exactly ONE new file (the workflow record)",
      len(new_files) == 1 and new_files[0].startswith(
          f"models/{MID}/workflows/workflow-"),
      f"new={new_files}")
EXPECTED_MODIFIED = {
    f"models/{MID}/weights.pt",        # M3 publication (atomic rewrite)
    f"models/{MID}/weights.sha256",    # its sidecar content hash
    f"models/{MID}/manifest.json",     # latest_checkpoint + updated_at
}
check("A3 only the three M3 publication files modified",
      set(modified) == EXPECTED_MODIFIED, f"modified={modified}")

print("\n".join(passed))
if failed:
    print("\n".join(failed))
    sys.exit(1)
print(f"\n{len(passed)}/{len(passed) + len(failed)} LIVE M60 CHECKS PASSED")
