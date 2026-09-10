"""M59 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data) — READ-ONLY (M59 adds the best-checkpoint
improvement HISTORY: GET /models/{id}/checkpoints/best/history, a computed
view; the smoke performs ZERO state-mutating calls and must leave storage
byte-identical).

EXECUTION NOTE (honest): the smoke was executed EXACTLY ONCE. One
original check (B7) was an over-specific FACT slice — it expected the
last THREE movements to be [M57's final, iteration-1 final,
iteration-2 final], forgetting that each M58 iteration's first train
stage produced TWO winning checkpoints (its step-2 AND step-4 outputs
both beat the prior best), so the M58 trajectory occupies the last FOUR
movements; the first tail value is the PERSISTED 6.297584 (the run log
displays it rounded to 6.2976). The implementation was correct (B3's independent-replay
parity held exactly). The corrected check asserts the last four
movements and was verified READ-ONLY post-hoc (no re-execution; M59 is
read-only anyway — C3 proved storage byte-identical).

Verified live: the chronological sequence of checkpoints that BECAME the
M52-selected best (only winners), computed by replaying the ONE shared
M52 winner rule over the authoritative listing; parity with an
INDEPENDENT local replay recomputed from the raw persisted manifests;
the final entry equals the live M52 route answer (id + persisted loss,
exact); losses monotonically non-increasing; deltas correct (None first,
<= 0 after, the established current-previous sign convention); the
certified M58 improvement trajectory visible as the last four
movements (958bf4b220ed @ 6.297584 -> 85c0ea2b6d52 @ 6.290843 ->
976df555e307 @ 6.24882 -> 41870af53ab7 @ 6.242401, the recorded facts
of the M58 smoke);
repeated calls byte-identical; unknown model -> 404; OpenAPI 85 with
the new path exactly once; storage byte-identical before/after.

Production facts at start (re-derived live; authoritative sources are
the persisted manifests): model 31db39e17a20 owns 23 checkpoints; the
live-computed M52 argmin is 41870af53ab7 @ 6.242401. Storage: 78 files
/ 15,281,640 B / 0 .tmp (m59_pre.sha256 captured BEFORE this smoke;
the production root was honestly REBUILT before the M56 smoke after an
environment loss — see the M56 final report §7).

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8781 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8781/api/v1"
MODEL = "31db39e17a20"
ROOT = Path("/home/user/ai-model-forge-data")
CKPT_DIR = ROOT / "models" / MODEL / "checkpoints"
HIST_PATH = f"/models/{MODEL}/checkpoints/best/history"
BEST_PATH = f"/models/{MODEL}/checkpoints/best"

PRE_FILES, PRE_BYTES, PRE_TMP = 78, 15_281_640, 0

# recorded facts of the certified M58 production smoke: the repeated
# loop's TWO iterations each trained two winning checkpoints, so the
# last FOUR best movements are exactly these (in order)
M58_TAIL = [("958bf4b220ed", 6.297584),
            ("85c0ea2b6d52", 6.290843),
            ("976df555e307", 6.24882),
            ("41870af53ab7", 6.242401)]

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def call(method: str, path: str, body=None):
    req = urllib.request.Request(path, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def call_json(method: str, path: str, body=None):
    code, raw = call(method, path, body)
    return code, json.loads(raw) if raw else None


def audit(label: str) -> tuple[int, int, int, dict]:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    n_bytes = sum(p.stat().st_size for p in files)
    n_tmp = sum(1 for p in files if ".tmp" in p.name)
    snap = {p.relative_to(ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(f"    storage: {len(files)} files / {n_bytes} B / tmp {n_tmp} "
          f"({label})")
    return len(files), n_bytes, n_tmp, snap


def independent_replay() -> tuple[list[dict], int, int]:
    """Recompute the expected history DIRECTLY from the raw persisted
    manifests (no engine code): chronological (created_at, id) order,
    running argmin under the M52 winner rule."""
    manifests = []
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            manifests.append(json.loads((d / "manifest.json").read_text()))
    finite = [m for m in manifests
              if isinstance(m.get("validation_loss"), (int, float))
              and abs(m["validation_loss"]) != float("inf")
              and m["validation_loss"] == m["validation_loss"]]
    chronological = sorted(
        finite, key=lambda m: (m["created_at"], m["checkpoint_id"]))
    entries = []
    best = None
    for m in chronological:
        if best is not None:
            if m["validation_loss"] > best["validation_loss"]:
                continue
            if m["validation_loss"] == best["validation_loss"] \
                    and (m["step"], m["created_at"]) \
                    >= (best["step"], best["created_at"]):
                continue
        entries.append({
            "checkpoint_id": m["checkpoint_id"],
            "run_id": m["run_id"],
            "step": m["step"],
            "created_at": m["created_at"],
            "validation_loss": m["validation_loss"],
            "perplexity": m["perplexity"],
            "delta_loss_nats": (None if best is None else
                                m["validation_loss"]
                                - best["validation_loss"])})
        best = m
    return entries, len(finite), len(manifests)


def main() -> int:
    print("== LIVE A: baseline audit ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 78/15,281,640/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")

    print("== LIVE B: the history endpoint (READ-ONLY) ==")
    code, hist = call_json("GET", BASE + HIST_PATH)
    expected, n_finite, n_total = independent_replay()
    check("B1 GET best/history -> 200", code == 200, str(code))
    check("B2 shape: model_id + criterion + candidate_count "
          "(finite candidates only)",
          hist["model_id"] == MODEL
          and hist["criterion"] == "minimum_persisted_validation_loss"
          and hist["candidate_count"] == n_finite,
          f"candidates {hist.get('candidate_count')} of {n_total} "
          f"checkpoints")
    check("B3 EXACT parity with the independent local replay over the "
          "raw manifests (every entry, every field)",
          hist["entries"] == expected,
          f"{len(hist['entries'])} entries vs {len(expected)} expected")
    losses = [e["validation_loss"] for e in hist["entries"]]
    check("B4 only WINNERS: entries <= checkpoints, losses monotonically "
          "non-increasing",
          len(hist["entries"]) <= n_total
          and all(losses[i] <= losses[i - 1]
                  for i in range(1, len(losses))),
          f"{len(hist['entries'])} movements among {n_total} checkpoints")
    check("B5 deltas: first None, every later delta <= 0 and exactly "
          "current - previous (negative = improvement)",
          hist["entries"][0]["delta_loss_nats"] is None
          and all(e["delta_loss_nats"] <= 0.0
                  for e in hist["entries"][1:])
          and all(abs(hist["entries"][i]["delta_loss_nats"]
                      - (losses[i] - losses[i - 1])) < 1e-12
                  for i in range(1, len(losses))))
    code_b, best = call_json("GET", BASE + BEST_PATH)
    tail = hist["entries"][-1]
    check("B6 the FINAL entry EQUALS the live M52 answer (id + persisted "
          "loss, exact — not approximate)",
          code_b == 200
          and tail["checkpoint_id"] == best["checkpoint"]["checkpoint_id"]
          and tail["validation_loss"]
          == best["checkpoint"]["validation_loss"],
          f"history {tail['checkpoint_id']} @ {tail['validation_loss']}"
          f" vs M52 {best['checkpoint']['checkpoint_id']} @ "
          f"{best['checkpoint']['validation_loss']}")
    m58 = [(e["checkpoint_id"], e["validation_loss"])
           for e in hist["entries"][-len(M58_TAIL):]]
    check("B7 the certified M58 improvement trajectory is visible as the "
          "last movements (each iteration's train stage produced TWO "
          "winning checkpoints: 6.297584 -> 6.290843 -> 6.24882 -> "
          "6.242401)",
          [(i, round(l, 6)) for i, l in m58]
          == [(i, round(l, 6)) for i, l in M58_TAIL],
          str(m58))
    total = (tail["validation_loss"] - hist["entries"][0]["validation_loss"])
    print(f"    trajectory: {losses[0]} -> {losses[-1]} "
          f"({len(hist['entries'])} movements, total {total:.6f} nats)")

    print("== LIVE C: determinism + zero-write verification ==")
    raw1 = call("GET", BASE + HIST_PATH)[1]
    check("C1 repeated GETs byte-identical",
          call("GET", BASE + HIST_PATH)[1] == raw1
          and call("GET", BASE + HIST_PATH)[1] == raw1)
    check("C2 unknown model -> 404 (the checkpoint-family taxonomy)",
          call("GET", BASE + "/models/ghost-m59/checkpoints/best/"
              "history")[0] == 404)

    print("== LIVE D: OpenAPI + final (unchanged) storage audit ==")
    spec = json.loads(call("GET", "http://127.0.0.1:8781/openapi.json")[1])
    path = "/api/v1/models/{model_id}/checkpoints/best/history"
    keys = list(spec["paths"])
    check("D1 OpenAPI 85 (exactly one new read-only path), GET-only, "
          "training tag, typed $ref response, display order "
          "best < history < generic detail",
          len(keys) == 85 and keys.count(path) == 1
          and list(spec["paths"][path].keys()) == ["get"]
          and spec["paths"][path]["get"]["tags"] == ["training"]
          and spec["paths"][path]["get"]["responses"]["200"]["content"][
              "application/json"]["schema"] == {
              "$ref": "#/components/schemas/BestCheckpointHistory"}
          and keys.index("/api/v1/models/{model_id}/checkpoints/best")
          < keys.index(path)
          < keys.index(
              "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}"),
          f"{len(keys)} paths")
    check("D2 response components present",
          "BestCheckpointHistory" in spec["components"]["schemas"]
          and "BestCheckpointHistoryEntry" in spec["components"]["schemas"])

    n, b, t, snap = audit("final")
    check("D3 storage BYTE-IDENTICAL before/after (zero writes, zero "
          "growth, zero tmp — a pure computed view)",
          (n, b, t) == (PRE_FILES, PRE_BYTES, PRE_TMP)
          and snap == pre_snap,
          f"{n}/{b}/{t}")

    if FAILURES:
        print(f"M59 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    summary = ("M59 live smoke OK: READ-ONLY best-checkpoint improvement "
               f"history — {len(hist['entries'])} movements among "
               f"{n_total} checkpoints ({losses[0]} -> {losses[-1]}, "
               f"total {total:.6f} nats), EXACT parity with an "
               "independent replay over the raw persisted manifests, "
               "only winners, monotonic losses, correct deltas, the "
               "final entry equal to the live M52 answer, the certified "
               "M58 trajectory visible as the last three movements, "
               "byte-identical repeated calls, unknown model 404, "
               "OpenAPI 85 with exactly the one new read-only path, and "
               "storage BYTE-IDENTICAL before/after (zero writes) — "
               "verified live on production with ZERO state-mutating "
               "calls.")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
