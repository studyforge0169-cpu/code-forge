"""M62 LIVE certification smoke (READ-ONLY — zero state-mutating calls).

Certifies the retention overview against the REBUILT production root
(see m62_production_rebuild.py for why the root was reconstructed):

    GET /models/{id}/checkpoints/retention
        vs
    independently computed facts (filesystem + record manifests)

Checks: shape/order/totals; the M52 best and M60 published checkpoints
are protected for exactly their reasons; one safe checkpoint (deletable,
zero blockers); one artifact-referenced checkpoint; deterministic
byte-identical repeat; storage byte-identical before/after (against
m62_pre.sha256); zero tmp.

NO deletion, NO publication, NO training, NO workflow execution — the
M62 smoke contains ONLY GET requests.

Usage: python smoke_m62_live.py http://127.0.0.1:<port>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
MID = "f4fea0ea6558"
DATA = "/home/user/ai-model-forge-data"
MODEL_DIR = f"{DATA}/models/{MID}"

# Certified pre-smoke facts of the rebuilt production root (captured
# read-only 2026-09-10 after the rebuild; see m62_pre.sha256):
BEST = "650965025ae1"            # == published (post-M60-style publish)
BEST_LOSS = 5.996844
N_CKPTS = 13
N_DELETABLE = 10
N_PROTECTED = 3
TOTAL_BYTES = 8197262            # checkpoint artifact sets only
RECLAIMABLE_BYTES = 6305615
N_HISTORY = 13

passed: list[str] = []
failed: list[str] = []


def check(section: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(
        f"{section}: {'PASS' if ok else 'FAIL ' + detail}")


def get(path: str):
    r = urllib.request.Request(BASE + path, method="GET")
    try:
        with urllib.request.urlopen(r) as resp:
            body = resp.read()
            return resp.status, json.loads(body), body
    except urllib.error.HTTPError as e:
        return e.code, None, b""


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------- independent fact computation
# (from the filesystem + record manifests ONLY — not via the app's
# analysis; this is the oracle the overview must agree with)
inv = {}
for line in open("/home/user/code-forge/m62_pre.sha256"):
    e = line.split(None, 1)
    if len(e) == 2:
        inv[e[1].strip().lstrip("./")] = e[0]

ck_dir = f"{MODEL_DIR}/checkpoints"
ck_ids = sorted(os.listdir(ck_dir))
sizes = {c: sum(os.path.getsize(f"{ck_dir}/{c}/{f}")
                for f in ("manifest.json", "weights.pt"))
         for c in ck_ids}
ck_records = {c: json.loads(open(f"{ck_dir}/{c}/manifest.json").read())
              for c in ck_ids}

blocked: set[str] = set()
block_reasons: dict[str, set[str]] = {}
def _block(cid, reason):
    blocked.add(cid)
    block_reasons.setdefault(cid, set()).add(reason)

manifest = json.loads(open(f"{MODEL_DIR}/manifest.json").read())
_block(manifest["latest_checkpoint"], "published")
if manifest.get("best_checkpoint"):
    _block(manifest["best_checkpoint"], "manifest_reference")
for w in os.listdir(f"{MODEL_DIR}/workflows"):
    wf = json.loads(open(f"{MODEL_DIR}/workflows/{w}/manifest.json").read())
    plan_blob = json.dumps(wf["plan"])
    for cid in ck_ids:
        if cid in plan_blob:
            _block(cid, "workflow_reference")
    for sr in wf["stages"]:
        art = sr.get("artifact") or {}
        if art.get("checkpoint_id"):
            _block(art["checkpoint_id"], "workflow_reference")
for family, reason in (("evaluations", "evaluation_reference"),
                       ("comparisons", "comparison_reference"),
                       ("gates", "gate_reference")):
    d = f"{MODEL_DIR}/{family}"
    for r in os.listdir(d):
        rec = json.dumps(json.loads(open(f"{d}/{r}/manifest.json").read()))
        for cid in ck_ids:
            if cid in rec:
                _block(cid, reason)
# the M52 best: independent replay (min persisted validation_loss,
# canonical (step, created_at) tie-break)
best_id = min(ck_records,
              key=lambda c: (ck_records[c]["validation_loss"],
                             ck_records[c]["step"],
                             ck_records[c]["created_at"])) \
    if any(v == v and abs(v) != float("inf")
           for v in (r["validation_loss"] for r in ck_records.values())) \
    else None
if best_id is not None:
    _block(best_id, "best")

independent_deletable = {c for c in ck_ids if c not in blocked}
independent_reclaimable = sum(sizes[c] for c in independent_deletable)

# ------------------------------------------------------------------ prelude
st, model, _ = get(f"/api/v1/models/{MID}")
check("P1 model readable", st == 200)
check("P2 published == certified best (post-publish state)",
      st == 200 and model["latest_checkpoint"] == BEST)

st, best, _ = get(f"/api/v1/models/{MID}/checkpoints/best")
check("P3 M52 best is the certified selection", st == 200
      and best["checkpoint"]["checkpoint_id"] == BEST
      and best["checkpoint"]["validation_loss"] == BEST_LOSS)
check("P3b independent best replay agrees",
      best_id == BEST)

st, hist, _ = get(f"/api/v1/models/{MID}/checkpoints/best/history")
check("P4 M59 history coherent", st == 200
      and hist["candidate_count"] == N_CKPTS
      and len(hist["entries"]) == N_HISTORY
      and hist["entries"][-1]["checkpoint_id"] == BEST)

# ------------------------------------------------------------- the overview
st, ov, ov_body = get(f"/api/v1/models/{MID}/checkpoints/retention")
check("X1 retention overview readable", st == 200)

check("V1 totals match certified facts",
      st == 200 and ov["model_id"] == MID
      and ov["total_checkpoints"] == N_CKPTS
      and ov["deletable_checkpoints"] == N_DELETABLE
      and ov["protected_checkpoints"] == N_PROTECTED
      and ov["total_checkpoint_bytes"] == TOTAL_BYTES
      and ov["reclaimable_checkpoint_bytes"] == RECLAIMABLE_BYTES,
      f"got {ov.get('total_checkpoints') if ov else None}/"
      f"{ov.get('deletable_checkpoints') if ov else None}/"
      f"{ov.get('protected_checkpoints') if ov else None}/"
      f"{ov.get('total_checkpoint_bytes') if ov else None}/"
      f"{ov.get('reclaimable_checkpoint_bytes') if ov else None}")

entries = ov["checkpoints"]
by_id = {e["checkpoint_id"]: e for e in entries}
# canonical (step, created_at) listing order + persisted identity parity
listing_order = sorted(ck_ids,
                       key=lambda c: (ck_records[c]["step"],
                                      ck_records[c]["created_at"]))
check("V2 canonical ordering + identity parity",
      [e["checkpoint_id"] for e in entries] == listing_order
      and all(e["run_id"] == ck_records[e["checkpoint_id"]]["run_id"]
              and e["step"] == ck_records[e["checkpoint_id"]]["step"]
              and e["validation_loss"]
              == ck_records[e["checkpoint_id"]]["validation_loss"]
              and e["files"] == 2
              and e["size_bytes"] == sizes[e["checkpoint_id"]]
              and e["integrity_verified"] is True
              for e in entries))

# deletable set == the independent oracle's set (exact equality)
check("V3 deletable set == independent oracle",
      {e["checkpoint_id"] for e in entries if e["deletable"]}
      == independent_deletable
      and all(e["blockers"] == [] for e in entries if e["deletable"]))
check("V4 reclaimable == sum of deletable sizes",
      sum(e["size_bytes"] for e in entries if e["deletable"])
      == independent_reclaimable == RECLAIMABLE_BYTES)

# best == published: protected, first blockers exactly [best, published]
bb = [b["reason"] for b in by_id[BEST]["blockers"]]
check("V5 best==published protected for exactly [best, published]",
      by_id[BEST]["deletable"] is False and bb[:2] == ["best", "published"],
      f"blockers={bb}")

# one SAFE checkpoint: deletable, zero blockers, size matches disk
safe = next(e for e in entries if e["deletable"])
check("V6 one safe checkpoint (deletable, zero blockers)",
      safe["blockers"] == [] and safe["size_bytes"] > 0
      and safe["size_bytes"] == sizes[safe["checkpoint_id"]])

# one ARTIFACT-REFERENCED checkpoint: protected with its record reasons
ref_ck = next(e["checkpoint_id"] for e in entries
              if not e["deletable"] and e["checkpoint_id"] != BEST
              and e["blockers"])
ref_reasons = {b["reason"] for b in by_id[ref_ck]["blockers"]}
check("V7 artifact-referenced checkpoint protected with record reasons",
      by_id[ref_ck]["deletable"] is False
      and ref_reasons <= block_reasons[ref_ck]
      and ref_reasons & {"workflow_reference", "evaluation_reference",
                         "comparison_reference", "gate_reference"},
      f"reasons={ref_reasons}")

# ---------------------------------------------------------------- determinism
st2, ov2, ov_body2 = get(f"/api/v1/models/{MID}/checkpoints/retention")
check("V8 deterministic byte-identical repeat",
      st2 == 200 and ov_body2 == ov_body)

# ------------------------------------------------------------------- audits
new_files, modified = [], []
for root, _dirs, files in os.walk(DATA):
    if root == f"{DATA}/tmp":
        continue
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, DATA)
        digest = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if rel not in inv:
            new_files.append(rel)
        elif inv[rel] != digest:
            modified.append(rel)
check("A1 storage byte-identical before/after (zero mutation)",
      new_files == [] and modified == [],
      f"new={new_files} modified={modified}")
check("A2 tmp clean",
      not os.path.exists(f"{DATA}/tmp")
      or os.listdir(f"{DATA}/tmp") == [])

print("\n".join(passed))
if failed:
    print("\n".join(failed))
    sys.exit(1)
print(f"\n{len(passed)}/{len(passed) + len(failed)} "
      f"LIVE M62 READ-ONLY CHECKS PASSED (zero mutations)")
